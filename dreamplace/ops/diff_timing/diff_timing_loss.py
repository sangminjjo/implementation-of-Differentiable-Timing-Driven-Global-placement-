import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from collections import defaultdict
import logging
import dreamplace.ops.timing.timing_cpp as timing_cpp
import json
import os
import time

try:
    import dreamplace.ops.diff_timing.diff_timing_cpp as diff_timing_cpp
    _RC_PROP_CPP = True
except ImportError:
    _RC_PROP_CPP = False

cell_propagation_cuda = None
try:
    from dreamplace.ops.diff_timing.cell_prop import cell_propagation_cuda
    _CELL_PROP_CUDA = True
except ImportError:
    _CELL_PROP_CUDA = False

_rc_prop_cuda_forward = None
try:
    from dreamplace.ops.diff_timing.rc_prop_cuda import rc_prop_cuda_forward as _rc_prop_cuda_forward
    _RC_PROP_CUDA = True
except ImportError:
    _RC_PROP_CUDA = False

_compute_timing_loss_cuda = None
try:
    from dreamplace.ops.diff_timing.timing_loss_cuda import compute_timing_loss_cuda as _compute_timing_loss_cuda
except ImportError:
    pass

_level_propagation = None
try:
    from dreamplace.ops.diff_timing.level_prop_cuda import level_propagation as _level_propagation
except ImportError:
    pass


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_tensor(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    torch.save(tensor.cpu(), path)

def parse_sdc_clock_period(sdc_path):
    '''sdc file에서 create_clock 명령어의 -period값과 클럭 소스 포트명을 parsing'''
    if not sdc_path:
        return 0.0, None

    period = 0.0
    clock_port = None
    try:
        with open(sdc_path, 'r') as f:
            for line in f:
                if 'create_clock' in line:
                    m = re.search(r'-period\s+([\d\.]+)', line)
                    if m:
                        period = float(m.group(1))
                    m2 = re.search(r'\[get_ports\s+\{?(\S+?)\}?\]', line)
                    if m2:
                        clock_port = m2.group(1)
                    break
    except Exception as e:
        logging.warning(f"Failed to read SDC file: {e}. Using default period 0.0")

    return period, clock_port

def extract_named_blocks(text, keyword):
    pattern = re.finditer(rf'{keyword}\s*\(\s*([^)]*)\s*\)\s*\{{', text)
    
    results = []
    for match in pattern:
        # [수정 1] name에 쌍따옴표가 포함될 수 있으므로 .strip('"') 추가
        name = match.group(1).strip().strip('"')
        start = match.end()

        depth = 1
        i = start
        
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1

        body = text[start:i-1]
        results.append((name, body))
    
    return results


def parse_liberty_nldm(lib_path):
    with open(lib_path, 'r') as f:
        content = f.read()

    # template parse
    #########################################
    templates = {}
    template_blocks = extract_named_blocks(content, "lu_table_template")

    for t_name, body in template_blocks:
        var1_match = re.search(r'variable_1\s*:\s*(\w+)', body)
        var2_match = re.search(r'variable_2\s*:\s*(\w+)', body)

        idx1_match = re.search(r'index_1\s*\(\s*"([^"]+)"', body)
        idx2_match = re.search(r'index_2\s*\(\s*"([^"]+)"', body)
        
        if not (var1_match and var2_match and idx1_match and idx2_match):
            continue
            
        is_var1_slew = 'transition' in var1_match.group(1)

        idx1 = [float(x) for x in idx1_match.group(1).split(',')]
        idx2 = [float(x) for x in idx2_match.group(1).split(',')]

        templates[t_name] = {
            "slew_axis": idx1 if is_var1_slew else idx2,
            "cap_axis": idx2 if is_var1_slew else idx1,
            "is_var1_slew": is_var1_slew
        }

    # cell parse
    ####################################################
    cell_luts = {}
    pin_info = {}
    
    cell_blocks = content.split('cell (')
    for block in cell_blocks[1:]:
        cell_name = block.split(')')[0].strip('"')
        cell_luts[cell_name] = {}
        pin_info[cell_name] = {}
        
        # pin parse
        pin_blocks = block.split('pin (')
        for p_block in pin_blocks[1:]:
            pin_name = p_block.split(')')[0].strip('"')
            pin_info[cell_name][pin_name] = {}

            dir_match = re.search(r'direction\s*:\s*(\w+)', p_block)
            cap_match = re.search(r'capacitance\s*:\s*([\d\.]+)', p_block)
            max_cap_match = re.search(r'max_capacitance\s*:\s*([\d\.]+)', p_block)
            max_tran_match = re.search(r'max_transition\s*:\s*([\d\.]+)', p_block)
            is_clock_match = re.search(r'clock\s*:\s*(true|false)', p_block)

            pin_info[cell_name][pin_name] = {
                "direction": dir_match.group(1) if dir_match else None,
                "cap": float(cap_match.group(1)) if cap_match else 0.0,
                "max_cap": float(max_cap_match.group(1)) if max_cap_match else None,
                "max_transition": float(max_tran_match.group(1)) if max_tran_match else None,
                "is_clock": True if is_clock_match and is_clock_match.group(1) == "true" else False
            }
            
            # timing() 블록 파싱
            timing_blocks = extract_named_blocks(p_block, "timing")

            for t_name, t_block in timing_blocks:

                rel_match = re.search(r'related_pin\s*:\s*"?([^";\s]+)"?', t_block)
                if not rel_match: continue
                in_pin = rel_match.group(1).strip()
                
                arc_data = {}
                
                sense_match = re.search(r'timing_sense\s*:\s*(\w+)', t_block)
                type_match = re.search(r'timing_type\s*:\s*(\w+)', t_block)
                
                arc_data["timing_sense"] = sense_match.group(1) if sense_match else "non_unate"
                arc_data["timing_type"] = type_match.group(1) if type_match else "combinational"

                # table parser
                def extract_table(keyword):
                    blocks = extract_named_blocks(t_block, keyword)
                    
                    if not blocks:
                        return None
                    
                    tb_name, body = blocks[0]

                    # scalar 테이블 (macro block output arcs, DFF setup/hold 등)
                    if tb_name == "scalar":
                        val_match = re.search(r'values\s*\(\s*"?\s*([\d\.\-eE+]+)\s*"?\s*\)', body)
                        if not val_match:
                            return None
                        v = float(val_match.group(1))
                        # Use wide 2-point axes so bilinear weights stay in [0,1] for any
                        # realistic slew/cap value, avoiding float32 catastrophic cancellation
                        # that occurs when both axis points are identical (wx → ±1e10).
                        return {"matrix": [[v, v], [v, v]], "slew_axis": [-1e6, 1e6], "cap_axis": [-1e6, 1e6]}

                    if tb_name not in templates:
                        return None

                    idx1_match = re.search(r'index_1\s*\(\s*"([^"]+)"', body)
                    idx2_match = re.search(r'index_2\s*\(\s*"([^"]+)"', body)

                    if idx1_match and idx2_match:
                        idx1 = [float(x) for x in idx1_match.group(1).split(',')]
                        idx2 = [float(x) for x in idx2_match.group(1).split(',')]

                        # [수정 2] t_name(빈 문자열) 대신 tb_name(테이블 이름) 사용
                        if templates[tb_name]["is_var1_slew"]:
                            slew_axis = idx1
                            cap_axis = idx2
                        else:
                            slew_axis = idx2
                            cap_axis = idx1
                    else:
                        # [수정 3] 여기서도 tb_name으로 교체
                        slew_axis = templates[tb_name]["slew_axis"]
                        cap_axis = templates[tb_name]["cap_axis"]

                    val_match = re.search(r'values\s*\(\s*\\?\s*(.*?)\s*\)', body, re.DOTALL)
                    if not val_match:
                        return None

                    val_body = val_match.group(1)
                    rows = re.findall(r'"([^"]+)"', val_body)

                    matrix = [[float(v) for v in row.split(',')] for row in rows]

                    #[수정 4] 여기서도 tb_name으로 교체
                    if not templates[tb_name]["is_var1_slew"]:
                        matrix = list(map(list, zip(*matrix)))

                    return {
                        "matrix": matrix,
                        "slew_axis": slew_axis,
                        "cap_axis": cap_axis
                    }

                arc_data["cell_fall"] = extract_table("cell_fall")
                arc_data["cell_rise"] = extract_table("cell_rise")

                arc_data["fall_transition"] = extract_table("fall_transition")
                arc_data["rise_transition"] = extract_table("rise_transition")

                arc_data["rise_constraint"] = extract_table("rise_constraint")
                arc_data["fall_constraint"] = extract_table("fall_constraint")

                if any(v is not None for k, v in arc_data.items() if "cell_" in k or "_transition" in k or "_constraint" in k):
                    if (in_pin, pin_name) not in cell_luts[cell_name]:
                        cell_luts[cell_name][(in_pin, pin_name)] = []
                    
                    cell_luts[cell_name][(in_pin, pin_name)].append(arc_data)

    return cell_luts, pin_info

class DifferentiableTimingLoss(nn.Module):
    def __init__(self, num_pins, timing_arcs, constraint_arcs, endpoint_list, placedb, params, device):
        super(DifferentiableTimingLoss, self).__init__()
        if _RC_PROP_CPP:
            logging.info("DifferentiableTimingLoss: diff_timing_cpp (C++) loaded OK")
        else:
            logging.warning("DifferentiableTimingLoss: diff_timing_cpp NOT found, using Python fallback")
        self.device = device
        self.num_pins = num_pins
        self.gamma = 100 #smoothing hyperparameter for LSE (gamma scheduling 대상)
        self.timing_growth_factor = getattr(params, 'timing_growth_factor', 1.01)
        self.placedb = placedb

        self.timing_arcs = timing_arcs
        self.constraint_arcs = constraint_arcs
        self.endpoint_list = endpoint_list

        # pos는 DreamPlace 내부 단위 (site units)
        # scale_factor = 1 / site_width_DBU, def_unit = DBU/µm
        # site → µm 변환: dx_µm = dx_sites * site_width_DBU / def_unit
        #                        = dx_sites / (scale_factor * def_unit)
        def_unit = float(placedb.rawdb.defUnit())
        self.pos_to_um = 1.0 / (params.scale_factor * def_unit) #0.19um/pos
        logging.info("DiffTiming pos_to_um=%.6f µm/site (scale_factor=%.6f, def_unit=%.0f)", self.pos_to_um, params.scale_factor, def_unit)

        # sdc file parsing -> clock period
        sdc_path = getattr(params, 'sdc_input', None)
        self.clock_period, self.clock_port_name = parse_sdc_clock_period(sdc_path)
        logging.info(f"Parsed Clock Period from SDC: {self.clock_period}, clock port: {self.clock_port_name}")

        # 0. liberty 파일 parsing  (캐시 가능한 모든 초기화 포함)
        lib_path = params.late_lib_input  # max analysis만.

        # ── 캐시 키 계산 ───────────────────────────────────────────────────
        import hashlib
        lib_mtime = os.path.getmtime(lib_path)
        cache_key_str = f"{lib_path}|{lib_mtime}|{num_pins}|{len(timing_arcs)}"
        cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()[:12]
        cache_dir  = "cache"
        cache_path = os.path.join(cache_dir, f"diff_timing_{cache_hash}.pt")

        if os.path.exists(cache_path):
            logging.info(f"[Cache] Loading precomputed timing init from {cache_path}")
            ck = torch.load(cache_path, map_location="cpu")
            self.lut_slew_axis      = ck["lut_slew_axis"].double().to(self.device)
            self.lut_cap_axis       = ck["lut_cap_axis"].double().to(self.device)
            self.lut_cell_rise      = ck["lut_cell_rise"].double().to(self.device)
            self.lut_cell_fall      = ck["lut_cell_fall"].double().to(self.device)
            self.lut_rise_tran      = ck["lut_rise_tran"].double().to(self.device)
            self.lut_fall_tran      = ck["lut_fall_tran"].double().to(self.device)
            self.lut_sense          = ck["lut_sense"].to(self.device)
            self.pin_caps_tensor    = ck["pin_caps_tensor"].double().to(self.device)
            self.cell_src_matrix    = ck["cell_src_matrix"].to(self.device)
            self.cell_arc_id_matrix = ck["cell_arc_id_matrix"].to(self.device)
            self.cell_fanin_mask    = ck["cell_fanin_mask"].to(self.device)
            self.cell_arcs          = ck["cell_arcs"]
            self.net_arcs           = ck["net_arcs"]
            self.pin_levels         = [t.to(self.device) for t in ck["pin_levels"]]
            self.endpoints          = ck["endpoints"].to(self.device)
            self.endpoint_setup_times = ck["endpoint_setup_times"].double().to(self.device)
            self._init_net_matrices(placedb, params)
            self._init_capture_ck_mapping(placedb)
            # PO output load caps are not stored in cache — reapply here
            po_pins_cached = self.endpoints[~self.has_capture_ck]
            self.pin_caps_tensor[po_pins_cached] = 4.0
            self.flat_level_pins, self.level_ptr = self._prepare_level_tensors()
            logging.info(f"[Cache] Loaded. Circuit depth: {len(self.pin_levels)}, "
                         f"Endpoints: {len(self.endpoints)}")
            self._debug_check_level_alternation()
            return

        logging.info("Parsing Liberty NLDM LUTs in Python..")
        cell_luts, parsed_pin_info = parse_liberty_nldm(lib_path)

        # lib parsing DEBUG - 전체 저장
        full_cell_luts = {}
        for c, arcs in cell_luts.items():
            full_cell_luts[c] = {}
            for (src, dst), v in arcs.items():
                full_cell_luts[c][f"{src}->{dst}"] = v
        save_json(full_cell_luts, "debug/cell_luts_full.json")
        save_json(parsed_pin_info, "debug/pin_info_full.json")

        #cell arc의 lut만 unique_luts에 저장하고, 해당 lut의 id를 lut_dict_to_id에 저장
        unique_luts = []
        lut_dict_to_id = {}

        for cell, arcs in cell_luts.items():
            for arc_pins, data_list in arcs.items():
                data = None
                for d in data_list:
                    if d.get("cell_rise") or d.get("cell_fall"):
                        data = d
                        break
                if data is None:
                    continue
                lut_dict_to_id[(cell, arc_pins[0], arc_pins[1])] = len(unique_luts)
                unique_luts.append(data)
                
        num_unique_luts = len(unique_luts)

        #debug
        save_json(unique_luts, "debug/unique_luts_full.json")

        lut_dict_to_id_serializable = {f"{cell}:{src}->{dst}": idx for (cell, src, dst), idx in lut_dict_to_id.items()}
        save_json(lut_dict_to_id_serializable, "debug/lut_dict_to_id_full.json")
        
        def get_axis_len(d, axis_name):
            for k in ["cell_fall", "cell_rise", "fall_transition", "rise_transition"]:
                if d.get(k) is not None: return len(d[k][axis_name])
            return 7
        
        max_slew_pts = max([get_axis_len(d, 'slew_axis') for d in unique_luts]) if unique_luts else 8
        max_cap_pts  = max([get_axis_len(d, 'cap_axis') for d in unique_luts]) if unique_luts else 7
        logging.info(f"[DEBUG] max_slew_pts: {max_slew_pts}, max_cap_pts: {max_cap_pts}")



        self.lut_slew_axis = torch.zeros((num_unique_luts, max_slew_pts), dtype=torch.float64, device=self.device)
        self.lut_cap_axis  = torch.zeros((num_unique_luts, max_cap_pts),  dtype=torch.float64, device=self.device)
        self.lut_cell_rise = torch.zeros((num_unique_luts, max_slew_pts, max_cap_pts), dtype=torch.float64, device=self.device)
        self.lut_cell_fall = torch.zeros((num_unique_luts, max_slew_pts, max_cap_pts), dtype=torch.float64, device=self.device)
        self.lut_rise_tran = torch.zeros((num_unique_luts, max_slew_pts, max_cap_pts), dtype=torch.float64, device=self.device)
        self.lut_fall_tran = torch.zeros((num_unique_luts, max_slew_pts, max_cap_pts), dtype=torch.float64, device=self.device)
        self.lut_sense     = torch.zeros(num_unique_luts, dtype=torch.int32, device=self.device)

        sense_map = {"positive_unate": 0, "negative_unate": 1, "non_unate": 2}
        
        for i, data in enumerate(unique_luts):
            self.lut_sense[i] = sense_map.get(data.get("timing_sense", "non_unate"), 2)

            # 축 저장 (임의로 유효한 테이블 1개를 골라 축 정보 가져옴)
            valid_table = data.get("cell_fall") or data.get("cell_rise")
            if valid_table:
                nx, ny = len(valid_table['slew_axis']), len(valid_table['cap_axis'])
                self.lut_slew_axis[i, :nx] = torch.tensor(valid_table['slew_axis'], device=self.device)
                self.lut_cap_axis[i, :ny]  = torch.tensor(valid_table['cap_axis'], device=self.device)
                
                # 안전하게 매트릭스 추출 함수
                def get_mat(key):
                    if data.get(key) is not None: return torch.tensor(data[key]['matrix'], device=self.device)
                    return torch.zeros((nx, ny), device=self.device)

                self.lut_cell_rise[i, :nx, :ny] = get_mat("cell_rise")
                self.lut_cell_fall[i, :nx, :ny] = get_mat("cell_fall")
                self.lut_rise_tran[i, :nx, :ny] = get_mat("rise_transition")
                self.lut_fall_tran[i, :nx, :ny] = get_mat("fall_transition")


                # 빈 공간 패딩
                if nx < max_slew_pts:
                    self.lut_slew_axis[i, nx:] = self.lut_slew_axis[i, nx-1] + 0.1
                    for mat in[self.lut_cell_rise, self.lut_cell_fall, self.lut_rise_tran, self.lut_fall_tran]:
                        mat[i, nx:, :] = mat[i, nx-1:nx, :]
                if ny < max_cap_pts:
                    self.lut_cap_axis[i, ny:] = self.lut_cap_axis[i, ny-1] + 0.1
                    for mat in[self.lut_cell_rise, self.lut_cell_fall, self.lut_rise_tran, self.lut_fall_tran]:
                        mat[i, :, ny:] = mat[i, :, ny-1:ny]
        
        #debug
        with open("debug/lut_slew_axis_full.json", "w") as f:
            json.dump(self.lut_slew_axis.cpu().tolist(), f, indent=2)
        with open("debug/lut_cap_axis_full.json", "w") as f:
            json.dump(self.lut_cap_axis.cpu().tolist(), f, indent=2)
        with open("debug/lut_cell_rise_full.json", "w") as f:
            json.dump(self.lut_cell_rise.cpu().tolist(), f, indent=2)
        with open("debug/lut_cell_fall_full.json", "w") as f:
            json.dump(self.lut_cell_fall.cpu().tolist(), f, indent=2)
        with open("debug/lut_rise_tran_full.json", "w") as f:
            json.dump(self.lut_rise_tran.cpu().tolist(), f, indent=2)
        with open("debug/lut_fall_tran_full.json", "w") as f:
            json.dump(self.lut_fall_tran.cpu().tolist(), f, indent=2)


        self.cell_arcs = []
        self.net_arcs = []

        #init cell matrices, cell arc를 lut delay 모델로 변환, pin_info에는 input pin capacitance 있음
        self._init_cell_matrices(placedb, lut_dict_to_id, parsed_pin_info)

        debug_cell_map = {
            int(dst): [(int(src), int(arc_id)) for src, arc_id in srcs]
            for dst, srcs in self.cell_dst_to_srcs.items()
        }

        save_json(debug_cell_map, "debug/cell_arc_mapping.json")


        #pin levelization
        self.pin_levels = self._compute_pin_levelization()

        levels_cpu = [lvl.cpu().tolist() for lvl in self.pin_levels]
        save_json(levels_cpu, "debug/pin_levels.json")

        self._debug_check_level_alternation()

        logging.info(f"[DEBUG] First level size: {len(levels_cpu[0]) if levels_cpu else 0}")
        logging.info(f"Levelization complete. Circuit depth(Max Levels): {len(self.pin_levels)}")
        logging.info(f"Numer of Endpoints identified: {len(self.endpoints)}")

        #setup time 계산
        self._compute_setup_times(placedb, cell_luts)


        #init net matrices
        self._init_net_matrices(placedb, params)

        # ── 캐시 저장 ───────────────────────────────────────────────────────
        try:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save({
                "lut_slew_axis":      self.lut_slew_axis.cpu(),
                "lut_cap_axis":       self.lut_cap_axis.cpu(),
                "lut_cell_rise":      self.lut_cell_rise.cpu(),
                "lut_cell_fall":      self.lut_cell_fall.cpu(),
                "lut_rise_tran":      self.lut_rise_tran.cpu(),
                "lut_fall_tran":      self.lut_fall_tran.cpu(),
                "lut_sense":          self.lut_sense.cpu(),
                "pin_caps_tensor":    self.pin_caps_tensor.cpu(),
                "cell_src_matrix":    self.cell_src_matrix.cpu(),
                "cell_arc_id_matrix": self.cell_arc_id_matrix.cpu(),
                "cell_fanin_mask":    self.cell_fanin_mask.cpu(),
                "cell_arcs":          self.cell_arcs,
                "net_arcs":           self.net_arcs,
                "pin_levels":         [t.cpu() for t in self.pin_levels],
                "endpoints":          self.endpoints.cpu(),
                "endpoint_setup_times": self.endpoint_setup_times.cpu(),
            }, cache_path)
            logging.info(f"[Cache] Saved timing init to {cache_path}")
        except Exception as e:
            logging.warning(f"[Cache] Failed to save cache: {e}")


        self._init_capture_ck_mapping(placedb)

        #PO output load capacitance 추가(sdc file에 명시되어 있는 값으로)
        po_mask = ~self.has_capture_ck #endpoint에서 D pin이 아닌 pin은 primary output
        po_pins = self.endpoints[po_mask]

        self.pin_caps_tensor[po_pins] = 4.0

        # __init__ 마지막에 호출
        self.flat_level_pins, self.level_ptr = self._prepare_level_tensors()

    def _prepare_level_tensors(self):
        # 모든 레벨의 핀을 하나로 합침
        flat_level_pins = torch.cat(self.pin_levels).to(torch.int32).to(self.device)
        
        # 각 레벨이 어디서 시작하고 끝나는지 인덱스 저장 (Pointer)
        level_ptr = [0]
        curr = 0
        for lvl in self.pin_levels:
            curr += len(lvl)
            level_ptr.append(curr)
        level_ptr = torch.tensor(level_ptr, dtype=torch.int32, device=self.device)
        
        return flat_level_pins, level_ptr

        

    def _debug_check_level_alternation(self):
        """
        level L → level L+1 사이의 arc가 올바른 type인지 검증.
          - L 짝수 (0, 2, 4, ...) → L+1 : net arc 여야 함
          - L 홀수 (1, 3, 5, ...) → L+1 : cell arc 여야 함
        """
        # 핀 → level 역인덱스
        pin_to_level = {}
        for lvl_idx, level in enumerate(self.pin_levels):
            for p in level.cpu().tolist():
                pin_to_level[p] = lvl_idx

        results = []
        total_violations = 0

        # net arc 검증: src level 짝수(even) → dst level 홀수(odd) 이어야 함
        # (multi-fanin으로 gap > 1이 될 수 있으므로 parity만 확인)
        net_violations = []
        for src, dst in self.net_arcs:
            src_lvl = pin_to_level.get(src, -1)
            dst_lvl = pin_to_level.get(dst, -1)
            if src_lvl % 2 != 0 or dst_lvl % 2 != 1:
                net_violations.append({"src": src, "dst": dst, "src_lvl": src_lvl, "dst_lvl": dst_lvl})
        results.append({
            "arc_type": "net",
            "total": len(self.net_arcs),
            "violations": len(net_violations),
            "sample": net_violations[:10],
        })
        total_violations += len(net_violations)

        # cell arc 검증: src level 홀수(odd) → dst level 짝수(even) 이어야 함
        # (multi-fanin으로 gap > 1이 될 수 있으므로 parity만 확인)
        cell_violations = []
        for src, dst in self.cell_arcs:
            src_lvl = pin_to_level.get(src, -1)
            dst_lvl = pin_to_level.get(dst, -1)
            if src_lvl % 2 != 1 or dst_lvl % 2 != 0:
                cell_violations.append({"src": src, "dst": dst, "src_lvl": src_lvl, "dst_lvl": dst_lvl})
        results.append({
            "arc_type": "cell",
            "total": len(self.cell_arcs),
            "violations": len(cell_violations),
            "sample": cell_violations[:10],
        })
        total_violations += len(cell_violations)

        save_json(results, "debug/level_alternation_check.json")
        if total_violations == 0:
            logging.info("[LevelCheck] PASS - net/cell arcs alternate correctly across all levels")
        else:
            logging.warning(f"[LevelCheck] FAIL - {total_violations} violations (net:{len(net_violations)}, cell:{len(cell_violations)}). See debug/level_alternation_check.json")


    def _init_capture_ck_mapping(self, placedb):
        '''Constraint Arc를 이용하여 Endpoint(D pin)에 대한 Clock 핀을 정확히 매핑'''
        
        # constraint_arcs는 (ck_pin_id, d_pin_id) 형태의 튜플 리스트입니다.
        # 이를 딕셔너리로 만들어 D핀을 키로, Clock핀을 값으로 매핑합니다.
        d_to_ck_map = {d_pin: ck_pin for ck_pin, d_pin in self.constraint_arcs}

        capture_ck_list = []
        for ep_pin in self.endpoints.cpu().tolist():
            # 해당 Endpoint(D핀)으로 들어오는 Constraint arc의 Source(Clock핀)를 찾음
            ck_pin = d_to_ck_map.get(ep_pin, -1)
            capture_ck_list.append(ck_pin)

        self.capture_ck_pins = torch.tensor(capture_ck_list, dtype=torch.long, device=self.device)
        self.has_capture_ck = (self.capture_ck_pins != -1)
        
        logging.info(f"Capture Clock Mapping: {self.has_capture_ck.sum().item()} out of {len(self.endpoints)} endpoints have explicit clock pins.")

    def _compute_setup_times(self, placedb, cell_luts):
        """liberty cell_luts에서 endpoint별 setup time 추출"""

        d_to_ck_map = {d_pin: ck_pin for ck_pin, d_pin in self.constraint_arcs}
        setup_times = []

        for ep_pin in self.endpoints.cpu().tolist():
            ck_pin = d_to_ck_map.get(ep_pin, -1)
            setup_r, setup_f = 0.0, 0.0 #rise, fall setup
            
            if ck_pin != -1:
                d_node = int(placedb.pin2node_map[ep_pin])
                cell_type = placedb.instance_celltype.get(d_node)
                
                if cell_type and cell_type in cell_luts:
                    try:
                        d_pin_name = placedb.pin_names[ep_pin].split(b':')[-1].decode('utf-8')
                        ck_pin_name = placedb.pin_names[ck_pin].split(b':')[-1].decode('utf-8')
                        arcs = cell_luts[cell_type].get((ck_pin_name, d_pin_name), [])

                        for arc in arcs:
                            timing_type = arc.get("timing_type", "").lower()

                            if "setup" not in timing_type:
                                continue

                            if arc.get("rise_constraint") is not None:
                                mat = arc["rise_constraint"]["matrix"]
                                setup_r = mat[0][0]
                            if arc.get("fall_constraint") is not None:
                                mat = arc["fall_constraint"]["matrix"]
                                setup_f = mat[0][0]
                    except Exception as e:
                        pass

            setup_times.append([setup_r, setup_f])

        self.endpoint_setup_times = torch.tensor(setup_times, dtype=torch.float64, device=self.device)
        logging.info(f"Computed Setup Times. Max Rise: {self.endpoint_setup_times[:,0].max().item():.2f}, Max Fall: {self.endpoint_setup_times[:,1].max().item():.2f}")
                


    #self.cell_dst_to_srcs, self.pin_caps_tensor 다시 확인, PlaceDB.py 숙지 먼저 해야될듯

    def _init_cell_matrices(self, placedb, lut_dict_to_id, parsed_pin_info):
        self.cell_dst_to_srcs = defaultdict(list) #self.cell_dst_to_srcs[dst_pin_id].append((src_pin_id, arc_id))
        self.pin_caps_tensor = torch.zeros(self.num_pins, dtype=torch.float64, device=self.device)
        
        if not hasattr(placedb, 'instance_celltype') or not placedb.instance_celltype:
            raise ValueError("[ERROR] placedb.instance_celltype is empty!")

        # ------------------------------------------------------------------
        #[STEP 1] 한 번만 파싱해서 캐싱해두기 (String 연산 병목 완벽 제거)
        # 배열 인덱스(pin_id)로 즉시 (cell_type, pin_name)을 찾을 수 있게 만듭니다.
        # ------------------------------------------------------------------
        pin_info_cache =[None] * self.num_pins
        
        for pin_id, full_name_bytes in enumerate(placedb.pin_names): #placedb.pin_names[pin_id] = pin_name
            # pin_id -> node_id (정수) -> cell_type (문자열)
            node_id = int(placedb.pin2node_map[pin_id])
            cell_type = placedb.instance_celltype.get(node_id)
            
            if cell_type: # 이 핀이 Cell 내부 핀이라면
                # 딱 한 번만 디코딩 및 split 수행
                pin_name = full_name_bytes.split(b':')[-1].decode('utf-8')
                pin_info_cache[pin_id] = (cell_type, pin_name)
                

                # Load Cap도 이 때 미리 채워넣음
                if cell_type in parsed_pin_info and pin_name in parsed_pin_info[cell_type]:
                    self.pin_caps_tensor[pin_id] = parsed_pin_info[cell_type][pin_name]["cap"]

        with open("debug/pin_info_cache.json", "w") as f:
            json.dump(pin_info_cache, f, indent=2)
        

        pin_names_debug = [x.decode("utf-8") if isinstance(x, bytes) else x for x in placedb.pin_names.tolist()]
        
        with open("debug/pin_names.json", "w") as f:
            json.dump(pin_names_debug, f, indent=2)

        # ------------------------------------------------------------------
        # [STEP 2] dump_garph에서 추출한 timing_arcs 순회 (초고속 매핑)
        # ------------------------------------------------------------------
        
        for src_pin_id, dst_pin_id in self.timing_arcs: # 둘 다 정수(Integer)
            src_info = pin_info_cache[src_pin_id]
            dst_info = pin_info_cache[dst_pin_id]
            if src_info is None or dst_info is None:
                self.net_arcs.append((src_pin_id, dst_pin_id))
                continue

            src_cell_type, src_pin_name = src_info
            _, dst_pin_name = dst_info
            lut_key = (src_cell_type, src_pin_name, dst_pin_name)

            src_node = placedb.pin2node_map[src_pin_id]
            dst_node = placedb.pin2node_map[dst_pin_id]

            if src_node == dst_node:
                cell_type = src_cell_type
                
                # (cell_type, src_pin, dst_pin)으로 LUT 딕셔너리 O(1) 탐색
                lut_key = (cell_type, src_pin_name, dst_pin_name)
                
                if lut_key in lut_dict_to_id:
                    arc_id = lut_dict_to_id[lut_key]
                    self.cell_dst_to_srcs[dst_pin_id].append((src_pin_id, arc_id))
                    self.cell_arcs.append((src_pin_id, dst_pin_id))
                    continue
                else:
                    continue

            self.net_arcs.append((src_pin_id, dst_pin_id))


        # ------------------------------------------------------------------
        # [STEP 3] 텐서 행렬 변환 (기존 코드 유지)
        # ------------------------------------------------------------------
        max_fanin = max([len(srcs) for srcs in self.cell_dst_to_srcs.values()]) if self.cell_dst_to_srcs else 1
    
        self.cell_src_matrix = torch.zeros((self.num_pins, max_fanin), dtype=torch.int32, device=self.device)
        self.cell_arc_id_matrix = torch.zeros((self.num_pins, max_fanin), dtype=torch.int32, device=self.device)
        self.cell_fanin_mask = torch.zeros((self.num_pins, max_fanin), dtype=torch.bool, device=self.device)

        for dst, srcs in self.cell_dst_to_srcs.items():
            for i, (src, arc_id) in enumerate(srcs):
                self.cell_src_matrix[dst, i] = src
                self.cell_arc_id_matrix[dst, i] = arc_id
                self.cell_fanin_mask[dst, i] = True


    def _init_net_matrices(self, placedb, params):
        self.pin2node_map = torch.tensor(placedb.pin2node_map, dtype=torch.long, device=self.device)
        self.pin_offset_x = torch.tensor(placedb.pin_offset_x, dtype=torch.float32, device=self.device)
        self.pin_offset_y = torch.tensor(placedb.pin_offset_y, dtype=torch.float32, device=self.device)
        self.num_nodes = placedb.num_nodes

        self.net_driver_map = torch.arange(self.num_pins, dtype=torch.int32, device=self.device)
        if len(self.net_arcs) > 0:
            net_src_list, net_dst_list = zip(*self.net_arcs)
            # dst는 인덱스로 사용하므로 long, src는 int32(net_driver_map dtype과 일치)
            self.net_src_indices = torch.tensor(net_src_list, dtype=torch.int32, device=self.device)
            self.net_dst_indices = torch.tensor(net_dst_list, dtype=torch.long, device=self.device)
            self.net_driver_map[self.net_dst_indices] = self.net_src_indices
        else:
            self.net_src_indices = torch.empty(0, dtype=torch.int32, device=self.device)
            self.net_dst_indices = torch.empty(0, dtype=torch.long, device=self.device)

        self.unit_r = params.wire_resistance_per_micron #ohm/um
        self.unit_c = params.wire_capacitance_per_micron #F/um

        # Per-net driver pin array for FLUTE tree rooting
        net_driver_pins_cpu = torch.full((placedb.num_nets,), -1, dtype=torch.int32)
        pin2net = placedb.pin2net_map  # numpy array: pin_id -> net_id (post-sort)
        for src, _dst in self.net_arcs:
            net_id = int(pin2net[src])
            net_driver_pins_cpu[net_id] = src
        self.net_driver_pins_per_net = net_driver_pins_cpu  # stays on CPU for C++ call

        # Ideal clock net: SDC create_clock 포트에서 직접 구동되는 net의 sinks는
        # 칩 전체에 걸쳐 거대한 Elmore 임펄스를 만들 수 있으므로 wire delay/impulse = 0 처리.
        self.ideal_net_sink_mask = torch.zeros(self.num_pins, dtype=torch.bool, device=self.device)
        clock_port = getattr(self, 'clock_port_name', None)
        if clock_port:
            # pin_name2id_map에서 직접 clock source pin_id를 조회 (pin_names 형식 불확실 우회)
            clock_pin_id = None
            for k, v in placedb.pin_name2id_map.items():
                k_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                if k_str == clock_port:
                    clock_pin_id = int(v)
                    self.ideal_net_sink_mask[clock_pin_id] = True
                    break
            self.clock_pin_id = clock_pin_id  # used in forward() to zero initial slew
            logging.info(f"Ideal clock net: clock_port='{clock_port}', clock_pin_id={clock_pin_id}")
            if clock_pin_id is not None:
                ideal_sink_set = {int(dst) for src, dst in self.net_arcs if int(src) == clock_pin_id}
                logging.info(f"Ideal clock net: {len(ideal_sink_set)} sink pins found")
                if ideal_sink_set:
                    ideal_idx = torch.tensor(sorted(ideal_sink_set), dtype=torch.long, device=self.device)
                    self.ideal_net_sink_mask[ideal_idx] = True
                    logging.info(f"Ideal clock net: mask applied to {len(ideal_sink_set)} pins")
            else:
                logging.warning(f"Clock port '{clock_port}' not found in pin_name2id_map")

        self.flute_update_interval = 20
        self.tree_edge_src = None
        self.tree_edge_dst = None
        self.steiner_dep_x = None
        self.steiner_dep_y = None

        #debug
        with open("debug/net_driver_map.json", "w") as f:
            json.dump(self.net_driver_map.cpu().tolist(), f, indent=2)


    def _net_propagation(self, level_pins, AT, slew, pin_delays, pin_impulses_sq):
        level_pins = level_pins.long()
        driver_pins = self.net_driver_map[level_pins].long()
        
        # AT, slew의 Shape:[num_pins, 2] -> 0: Rise, 1: Fall
        pin_delays = pin_delays[level_pins].unsqueeze(1)  # [N, 1]
        pin_impulses_sq = pin_impulses_sq[level_pins].unsqueeze(1)
        
        AT[level_pins] = AT[driver_pins] + pin_delays #AT(v) = AT(u) + delay(v)
        eps = 1e-12
        driver_slew = slew[driver_pins]

        calculated_slew = torch.sqrt(driver_slew**2 + pin_impulses_sq + eps)

        slew[level_pins] = torch.where(driver_slew < 0.0, -calculated_slew, calculated_slew) #slew(v) = sqrt(slew(u)^2 + impulse(v)^2)

        return AT, slew

    ##############################################################################################
    def _cell_propagation(self, level_pins, AT, slew, pin_load_caps, gamma):
        level_pins = level_pins.long()
        src_matrix = self.cell_src_matrix[level_pins]      # [N, max_fanin], N = number of current level pins
        arc_ids = self.cell_arc_id_matrix[level_pins]      # [N, max_fanin]
        mask = self.cell_fanin_mask[level_pins]            # [N, max_fanin]
        
        if not mask.any(): return AT, slew

        # 0: Rise, 1: Fall
        in_AT_R = AT[src_matrix, 0]      # [N, max_fanin]
        in_AT_F = AT[src_matrix, 1]      #[N, max_fanin]
        in_slew_R = slew[src_matrix, 0]  #[N, max_fanin]
        in_slew_F = slew[src_matrix, 1]  # [N, max_fanin]
        out_cap = pin_load_caps[level_pins].unsqueeze(1) # [N, 1]

        # 4가지 LUT 호출을 2번으로 배칭 (kernel launch 횟수 절반)
        N = in_slew_R.shape[0]
        in_slew_cat = torch.cat([in_slew_R, in_slew_F], dim=0)  # [2N, max_fanin]
        arc_ids_2x  = torch.cat([arc_ids,   arc_ids],   dim=0)  # [2N, max_fanin]
        out_cap_2x  = torch.cat([out_cap,   out_cap],   dim=0)  #[2N, 1]

        d_rise_batch, s_rise_batch = self._differentiable_lut_interp_extrap(
            in_slew_cat, out_cap_2x, arc_ids_2x, self.lut_cell_rise, self.lut_rise_tran)
        d_RR, d_FR = d_rise_batch[:N], d_rise_batch[N:]
        s_RR, s_FR = s_rise_batch[:N], s_rise_batch[N:]

        d_fall_batch, s_fall_batch = self._differentiable_lut_interp_extrap(
            in_slew_cat, out_cap_2x, arc_ids_2x, self.lut_cell_fall, self.lut_fall_tran)
        d_RF, d_FF = d_fall_batch[:N], d_fall_batch[N:]
        s_RF, s_FF = s_fall_batch[:N], s_fall_batch[N:]

        # Unateness 마스크 생성
        sense = self.lut_sense[arc_ids] # 0: POS, 1: NEG, 2: NON
        is_pos = (sense == 0)
        is_neg = (sense == 1)
        is_non = (sense == 2)

        valid_RR_FF = is_pos | is_non
        valid_RF_FR = is_neg | is_non

        # AT, Slew 모두 동일하게 -1e9로 마스킹.
        # LSE에서 exp(-1e9/γ) ≈ 0이므로 무효 arc는 기여 없음 (NaN 없음).
        # 과거 주석 "확률 0 × -1e9 = NaN"은 weighted sum 방식일 때 이야기임.
        # LSE 방식에서는 exp(-1e9/γ) ≈ 0이기 때문에 NaN이 발생하지 않음.
        MIN_VAL = torch.full((), -1e9, dtype=AT.dtype, device=self.device)

        def apply_mask_at(val, valid_mask):
            return torch.where(mask & valid_mask, val, MIN_VAL)

        def apply_mask_slew(val, valid_mask):
            # 유효하지 않은 arc를 -1e9로 마스킹: exp(-1e9/γ) ≈ 0 → LSE에 기여 없음
            # 0으로 마스킹하면 exp(0/γ) = 1 → 무효 arc가 slew를 과대추정하게 됨
            return torch.where(mask & valid_mask, val, MIN_VAL)

        # Output Rise Arrival Time 후보들
        out_AT_RR = apply_mask_at(in_AT_R + d_RR, valid_RR_FF)
        out_AT_FR = apply_mask_at(in_AT_F + d_FR, valid_RF_FR)
        
        # Output Fall Arrival Time 후보들
        out_AT_FF = apply_mask_at(in_AT_F + d_FF, valid_RR_FF)
        out_AT_RF = apply_mask_at(in_AT_R + d_RF, valid_RF_FR)

        # Output Slew 후보들
        out_sl_RR = apply_mask_slew(s_RR, valid_RR_FF)
        out_sl_FR = apply_mask_slew(s_FR, valid_RF_FR)
        out_sl_FF = apply_mask_slew(s_FF, valid_RR_FF)
        out_sl_RF = apply_mask_slew(s_RF, valid_RF_FR)

        #[수정포인트 2] 텐서 평탄화 (cat 활용) : Shape -> [N, max_fanin * 2]
        # RR과 FR이 Rise 경쟁, FF와 RF가 Fall 경쟁
        AT_Rise_cands = torch.cat([out_AT_RR, out_AT_FR], dim=1)
        AT_Fall_cands = torch.cat([out_AT_FF, out_AT_RF], dim=1)
        
        sl_Rise_cands = torch.cat([out_sl_RR, out_sl_FR], dim=1)
        sl_Fall_cands = torch.cat([out_sl_FF, out_sl_RF], dim=1)

        # [수정포인트 3] AT는 기존대로 LogSumExp로 병합 (Max 모사)
        AT[level_pins, 0] = gamma * torch.logsumexp(AT_Rise_cands / gamma, dim=1)
        AT[level_pins, 1] = gamma * torch.logsumexp(AT_Fall_cands / gamma, dim=1)


        # 2. Slew 후보들과 가중치를 곱하여 Weighted Sum 연산
        slew[level_pins, 0] = gamma * torch.logsumexp(sl_Rise_cands / gamma, dim=1)
        slew[level_pins, 1] = gamma * torch.logsumexp(sl_Fall_cands / gamma, dim=1)

        return AT, slew
    

    def _differentiable_lut_interp_extrap(self, in_slew, out_cap, arc_ids, delay_grid_base, slew_grid_base):
        axes_slew = self.lut_slew_axis[arc_ids]
        axes_cap = self.lut_cap_axis[arc_ids]
        out_cap_exp = out_cap.expand_as(in_slew).contiguous()

         # 1. raw index 추출 (0 ~ max_len 반환)
        raw_idx_x = torch.searchsorted(axes_slew, in_slew.unsqueeze(-1)).squeeze(-1)
        raw_idx_y = torch.searchsorted(axes_cap, out_cap_exp.unsqueeze(-1)).squeeze(-1)

        # ========== [여기만 수정됨: 인덱스 Clamp] ==========
        # idx_x와 idx_y는 왼쪽(0) 지점을 의미하므로 최대 인덱스는 (길이 - 2) 여야 합니다. 
        # (길이 - 2)로 해야 나중에 idx_x + 1 을 할 때 배열을 벗어나지 않습니다.
        max_x = axes_slew.shape[2] - 2  # shape[2]=S (axis len), not shape[1]=max_fanin
        max_y = axes_cap.shape[2] - 2

        # min=0 : 0보다 작아져서 '-1(마지막 원소 참조)'이 되는 버그 완벽 차단 (Underflow시 외삽 보장)
        # max=... : 최대 인덱스를 벗어나서 Out-Of-Bounds 에러 나는 것 차단 (Overflow시 외삽 보장)
        idx_x = torch.clamp(raw_idx_x - 1, min=0, max=max_x)
        idx_y = torch.clamp(raw_idx_y - 1, min=0, max=max_y)
        # ===================================================

        x0 = torch.gather(axes_slew, 2, idx_x.unsqueeze(-1)).squeeze(-1)
        x1 = torch.gather(axes_slew, 2, (idx_x + 1).unsqueeze(-1)).squeeze(-1)
        y0 = torch.gather(axes_cap, 2, idx_y.unsqueeze(-1)).squeeze(-1)
        y1 = torch.gather(axes_cap, 2, (idx_y + 1).unsqueeze(-1)).squeeze(-1)

        wx = (in_slew - x0) / (x1 - x0 + 1e-9)
        wy = (out_cap_exp - y0) / (y1 - y0 + 1e-9)

        delay_grid = delay_grid_base[arc_ids]
        slew_grid = slew_grid_base[arc_ids]
        
        grid_y_size = delay_grid.shape[-1]
        delay_grid_flat = delay_grid.view(delay_grid.shape[0], delay_grid.shape[1], -1)
        slew_grid_flat = slew_grid.view(slew_grid.shape[0], slew_grid.shape[1], -1)

        def get_val(grid_flat, ix, iy):
            return torch.gather(grid_flat, 2, (ix * grid_y_size + iy).unsqueeze(-1)).squeeze(-1)

        v00_d, v01_d = get_val(delay_grid_flat, idx_x, idx_y), get_val(delay_grid_flat, idx_x, idx_y + 1)
        v10_d, v11_d = get_val(delay_grid_flat, idx_x+1, idx_y), get_val(delay_grid_flat, idx_x+1, idx_y + 1)
        
        v00_s, v01_s = get_val(slew_grid_flat, idx_x, idx_y), get_val(slew_grid_flat, idx_x, idx_y + 1)
        v10_s, v11_s = get_val(slew_grid_flat, idx_x+1, idx_y), get_val(slew_grid_flat, idx_x+1, idx_y + 1)

        w00 = (1 - wx) * (1 - wy)
        w01 = (1 - wx) * wy
        w10 = wx * (1 - wy)
        w11 = wx * wy

        arc_delay = w00 * v00_d + w01 * v01_d + w10 * v10_d + w11 * v11_d
        arc_slew = torch.relu(w00 * v00_s + w01 * v01_s + w10 * v10_s + w11 * v11_s)

        return arc_delay, arc_slew

    def _compute_pin_levelization(self):
        all_edges = self.timing_arcs
        adj_list = defaultdict(list)
        in_degree = [0] * self.num_pins

        for src, dst in all_edges:
            adj_list[src].append(dst)
            in_degree[dst] += 1

        
        self.endpoints = torch.tensor(self.endpoint_list, dtype=torch.long, device=self.device) #FF의 D pin, PO
        #debug endpoints
        with open("debug/endpoints.json", "w") as f:
            json.dump(self.endpoints.tolist(), f, indent=2)

        current_level_pins =[i for i in range(self.num_pins) if in_degree[i] == 0]
        levels =[]
        while current_level_pins:
            level_tensor = torch.tensor(current_level_pins, dtype=torch.long, device=self.device)
            levels.append(level_tensor)
            next_level_pins =[]
            for pin in current_level_pins:
                for neighbor in adj_list[pin]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_level_pins.append(neighbor)
            current_level_pins = next_level_pins
            
        return levels
    
    def _update_flute_trees(self, pos):
        logging.info("Updating Steiner Trees using FLUTE C++ Extension..")
        tensors = timing_cpp.extract_flute_topology( #fluate 에서 만들어지는 tree_edge는 cell arc 포함하지 않음. net에 대해서만 steiner point를 추가하고 edge 만듦
            self.placedb.num_nets, self.num_pins, pos.cpu().contiguous(),
            torch.tensor(self.placedb.flat_net2pin_map, dtype=torch.int32).contiguous(),
            torch.tensor(self.placedb.flat_net2pin_start_map, dtype=torch.int32).contiguous(),
            torch.tensor(self.placedb.pin2node_map, dtype=torch.int32).contiguous(),
            torch.tensor(self.placedb.pin_offset_x, dtype=pos.dtype).contiguous(),
            torch.tensor(self.placedb.pin_offset_y, dtype=pos.dtype).contiguous(),
            self.net_driver_pins_per_net.contiguous()
        )
        self.tree_edge_src = tensors[0].long().to(self.device) #steiner point 포함. shape: (num_edges, ), tree edge src의 의미가 맞도록 cpp 수정 완료
        self.tree_edge_dst = tensors[1].long().to(self.device)
        self.steiner_dep_x = tensors[3].long().to(self.device) #steiner point가 의존하는 pin의 id
        self.steiner_dep_y = tensors[4].long().to(self.device)

        #debug    
        # with open("debug/tree_edge_src.json", "w") as f:
        #     json.dump(self.tree_edge_src.tolist(), f, indent=2)
        # with open("debug/tree_edge_dst.json", "w") as f:
        #     json.dump(self.tree_edge_dst.tolist(), f, indent=2)
        # with open("debug/steiner_dep_x.json", "w") as f:
        #     json.dump(self.steiner_dep_x.tolist(), f, indent=2)
        # with open("debug/steiner_dep_y.json", "w") as f:
        #     json.dump(self.steiner_dep_y.tolist(), f, indent=2)
    
    def _compute_pin_positions(self, pos): # [units distance microns]
        node_x = pos[:self.num_nodes]
        node_y = pos[self.num_nodes:]
        pin_x = node_x[self.pin2node_map] + self.pin_offset_x
        pin_y = node_y[self.pin2node_map] + self.pin_offset_y
        return pin_x, pin_y
    
    #net의 delay, impulse, cap을 계산해서 pin에 할당.
    def _compute_differentiable_wire_rc(self, pin_x, pin_y):
        pin_x = pin_x.double()
        pin_y = pin_y.double()
        # pos 단위는 site (PlaceDB.scale()이 1/site_width_um을 곱했음)
        # site → µm 변환: dx_um = dx_sites * pos_to_um (= site_width_um, e.g. 0.19)
        # [수정] 엣지가 아예 없는 상황 방어
        if self.tree_edge_src is None or len(self.tree_edge_src) == 0:
            z = (pin_x * 0.0).sum() + torch.zeros(self.num_pins, device=self.device, dtype = pin_x.dtype)
            logging.info("Error in computing wire rc")
            return z, z, z

        if self.steiner_dep_x is not None and len(self.steiner_dep_x) > 0:
            #steiner dependency를 이용해 실제 좌표를 뽑아옴
            steiner_coords_x = pin_x[self.steiner_dep_x]
            steiner_coords_y = pin_y[self.steiner_dep_y]

            X = torch.cat([pin_x, steiner_coords_x]) #shape: (num_total_nodes, )
            Y = torch.cat([pin_y, steiner_coords_y])
        else:
            X, Y = pin_x, pin_y

        num_total_nodes = X.shape[0]

        dx = torch.abs(X[self.tree_edge_src] - X[self.tree_edge_dst]) * self.pos_to_um  # [µm]
        dy = torch.abs(Y[self.tree_edge_src] - Y[self.tree_edge_dst]) * self.pos_to_um
        dist = dx + dy  # Manhattan distance for rectilinear routing

        #debug
        #with open("debug/dx.json", "w") as f:
        #    json.dump(dx.tolist(), f, indent=2)

        edge_R   = dist * self.unit_r/1000  # [kΩ], total (impulse 계산용)

        # node_C: total dist 기반 wire cap + pin cap (Load는 R과 무관 - Eq.7a)
        edge_C = dist * self.unit_c * 1e15  # [fF]
        half_edge_C = edge_C * 0.5
        node_C = torch.zeros(num_total_nodes, device=self.device, dtype=pin_x.dtype)
        node_C.scatter_add_(0, self.tree_edge_src, half_edge_C)
        node_C.scatter_add_(0, self.tree_edge_dst, half_edge_C)
        node_C[:self.num_pins] += self.pin_caps_tensor  # [fF]

        src = self.tree_edge_src
        dst = self.tree_edge_dst
        MAX_DEPTH = 4

        pin_delays      = torch.zeros(self.num_pins, device=self.device, dtype=pin_x.dtype)
        pin_impulses_sq = torch.zeros(self.num_pins, device=self.device, dtype=pin_x.dtype)
        pin_capacitance = torch.zeros(self.num_pins, device=self.device, dtype=pin_x.dtype)

        if _RC_PROP_CUDA and node_C.is_cuda:
            # CUDA: 직접 작성한 scatter kernel 사용 (kernel launch 횟수 절감)
            pin_delays, pin_impulses_sq, pin_capacitance = _rc_prop_cuda_forward(node_C, edge_R, src, dst, self.num_pins, MAX_DEPTH)

            # ideal clock net sinks: wire delay/impulse = 0 (chip-spanning clock net은 무한대 임펄스 방지)
            if self.ideal_net_sink_mask.any():
                mask_float = (~self.ideal_net_sink_mask).to(pin_delays.dtype)
                pin_delays   = pin_delays   * mask_float
                pin_impulses_sq = pin_impulses_sq * mask_float
                pin_capacitance = pin_capacitance * mask_float

            pin_delays      = pin_delays[:self.num_pins]
            pin_impulses_sq    = pin_impulses_sq[:self.num_pins]  # actual impulse (ps), consistent with CUDA/CPP RC
            pin_capacitance = pin_capacitance[:self.num_pins]

        elif _RC_PROP_CPP:
            # delay
            # impulse: total R로 계산 (Beta는 x/y cross-term 때문에 선형 분리 불가)
            out_total = diff_timing_cpp.forward(node_C, edge_R, src, dst, self.num_pins, MAX_DEPTH)
            pin_delays = out_total[0]
            pin_impulses   = out_total[1]
            pin_capacitance = out_total[2]
        else:
            # Load 1회 계산 (R과 무관)
            load = node_C.clone()
            for _ in range(MAX_DEPTH):
                msg_sum = torch.zeros_like(load).scatter_add_(0, src, load[dst])
                load = node_C + msg_sum

            def _delay_pass(edge_R_dir):
                d = torch.zeros(num_total_nodes, device=self.device, dtype=pin_x.dtype)
                for _ in range(MAX_DEPTH):
                    new_d = d.clone()
                    new_d.scatter_(0, dst, d[src] + edge_R_dir * load[dst])
                    d = new_d
                return d
            

            delay = _delay_pass(edge_R)

            # impulse: total delay + total R로 계산
            base_ldelay = node_C * delay
            ldelay = base_ldelay.clone()
            for _ in range(MAX_DEPTH):
                msg_sum = torch.zeros_like(ldelay).scatter_add_(0, src, ldelay[dst])
                ldelay = base_ldelay + msg_sum

            beta = torch.zeros(num_total_nodes, device=self.device, dtype=pin_x.dtype)
            for _ in range(MAX_DEPTH):
                new_beta = beta.clone()
                new_beta.scatter_(0, dst, beta[src] + edge_R * ldelay[dst])
                beta = new_beta

            impulse_sq = 2.0 * beta - delay ** 2
            # impulse    = torch.sqrt(torch.clamp(impulse_sq, min=1e-12))

            pin_delays      = delay[:self.num_pins]
            pin_impulses_sq    = impulse_sq[:self.num_pins]  # actual impulse (ps), consistent with CUDA/CPP RC
            pin_capacitance = load[:self.num_pins]


        return pin_delays, pin_impulses_sq, pin_capacitance

    def forward(self, pos, iteration=0):
        if not hasattr(self, 'last_flute_iter'):
            self.last_flute_iter = -1

        if iteration >= 200:
            if iteration % self.flute_update_interval == 0:
                if self.last_flute_iter != iteration:
                    with torch.no_grad():
                        logging.info("Updating flute trees")
                        self._update_flute_trees(pos.detach())
                    self.last_flute_iter = iteration

    
            pin_pos_x, pin_pos_y = self._compute_pin_positions(pos)

            pin_delays, pin_impulses_sq, pin_capacitance = self._compute_differentiable_wire_rc(
                pin_pos_x, pin_pos_y
            )
            
            AT = torch.zeros((self.num_pins, 2), device=self.device, dtype=torch.float64)
            slew = torch.full((self.num_pins, 2), 10.0, device=self.device, dtype=torch.float64)
            # SDC set_input_transition 0.0 iccad_clk → clock port initial slew = 0 ps
            if getattr(self, 'clock_pin_id', None) is not None:
                slew[self.clock_pin_id, :] = 0.0

            if _level_propagation is not None:
                AT, slew = _level_propagation(
                    AT, slew, pin_delays, pin_impulses_sq, pin_capacitance, self)
            else:
                # Python fallback
                for level_idx, level_pins in enumerate(self.pin_levels):
                    if level_idx == 0:
                        continue
                    if level_idx % 2 == 1:
                        AT, slew = self._net_propagation(level_pins, AT, slew, pin_delays, pin_impulses_sq)
                    else:
                        if _CELL_PROP_CUDA and cell_propagation_cuda is not None:
                            AT, slew = cell_propagation_cuda(self, level_pins, AT, slew, pin_capacitance, self.gamma)
                        else:
                            AT, slew = self._cell_propagation(level_pins, AT, slew, pin_capacitance, self.gamma)
        
            # ── WNS / TNS penalty ──────────────────────────────────────────
            # AT clamp 제거: neg_slack_over_gamma clamp으로 충분히 overflow 방지됨.
            # clock_period + 80*gamma 상한은 gamma=0.001일 때 9500.08ps로 너무 타이트해
            # 실제 위반(>0.08ps)을 모두 잘라냄 → WNS/TNS가 0 근처로 붕괴됨.

            AT_for_loss = AT.clone()

            AT_for_loss[self.endpoints] += self.endpoint_setup_times

            # timing_loss_cuda.py handles CUDA/CPU dispatch and capture clock internally
            if _compute_timing_loss_cuda is not None:
                wns_penalty, tns_penalty = _compute_timing_loss_cuda(
                    AT_for_loss, self.endpoints, self.capture_ck_pins, self.has_capture_ck, float(self.clock_period), self.gamma)
            else:
                # True fallback: timing_loss_cuda.py itself failed to import
                endpoint_AT = AT_for_loss[self.endpoints]
                RAT = torch.full_like(endpoint_AT, self.clock_period)
                if self.has_capture_ck.any():
                    valid_ck_pins = self.capture_ck_pins[self.has_capture_ck]
                    clock_AT_rise = AT_for_loss[valid_ck_pins, 0].unsqueeze(1)
                    RAT[self.has_capture_ck] = self.clock_period + clock_AT_rise
                slacks = RAT - endpoint_AT                           # [E, 2]
                # softmin over rise/fall → [E] (미분가능)
                slacks_per_ep = -self.gamma * torch.logsumexp(-slacks / self.gamma, dim=1)
                neg_slack_over_gamma = -slacks_per_ep / self.gamma
                wns_penalty = self.gamma * torch.logsumexp(neg_slack_over_gamma, dim=0)
                tns_penalty = self.gamma * F.softplus(neg_slack_over_gamma).sum()


            tns_weight = 0.0001   * (self.timing_growth_factor ** iteration)
            wns_weight = 0.000001 * (self.timing_growth_factor ** iteration)


            timing_loss = wns_weight * wns_penalty + tns_weight * tns_penalty

            with torch.no_grad():
                endpoint_AT_dbg = AT_for_loss[self.endpoints]
                RAT_dbg = torch.full_like(endpoint_AT_dbg, self.clock_period)

                if self.has_capture_ck.any():
                    valid_ck_pins = self.capture_ck_pins[self.has_capture_ck]
                    clock_AT_rise = AT_for_loss[valid_ck_pins, 0].unsqueeze(1)
                    RAT_dbg[self.has_capture_ck] = self.clock_period + clock_AT_rise

                slacks_flat_dbg = (RAT_dbg - endpoint_AT_dbg).view(-1)
                # worst-of-rise/fall per endpoint (matches OpenTimer report_tns_elw)
                slacks_per_ep = (RAT_dbg - endpoint_AT_dbg).min(dim=1).values  # [E]
                wns_val = -slacks_per_ep.min().item()
                tns_val = -slacks_per_ep[slacks_per_ep < 0].sum().item()

                # 매 100 iteration마다 진단 로그
                if iteration % 100 == 0:
                    # 1. wire delay 분포
                    valid_wire = pin_delays[pin_delays > 0]
                    if len(valid_wire) > 0:
                        logging.info("[Diag] wire delay (ps) | max=%.1f, mean=%.1f, p99=%.1f",
                                    valid_wire.max().item(),
                                    valid_wire.mean().item(),
                                    valid_wire.quantile(0.99).item())
                    else:
                        logging.warning("[Diag] wire delay: all zero!")

                    # 2. endpoint AT 분포
                    ep_AT = AT[self.endpoints].view(-1)
                    logging.info("[Diag] endpoint AT (ps) | max=%.1f, mean=%.1f, p99=%.1f",
                                ep_AT.max().item(), ep_AT.mean().item(), ep_AT.quantile(0.99).item())

                    # 3. worst endpoint 상세
                    # slacks_flat_dbg = (clock_period - AT[endpoints]).view(-1), shape: [num_endpoints * 2]
                    # argmin은 flattened index → endpoint_idx = argmin // 2

                    worst_flat_idx = slacks_flat_dbg.argmin().item()
                    worst_ep_idx = worst_flat_idx // 2
                    worst_ep_pin = self.endpoints[worst_ep_idx].item()
                    worst_AT_R = AT[worst_ep_pin, 0].item()
                    worst_AT_F = AT[worst_ep_pin, 1].item()
                    worst_ck_AT = 0.0
                    if self.has_capture_ck[worst_ep_idx]:
                        worst_ck_pin = self.capture_ck_pins[worst_ep_idx]
                        worst_ck_AT = AT[worst_ck_pin, 0].item()

                        logging.info("[Diag] worst endpoint pin_id=%d | Data_AT(R,F)=(%.1f, %.1f) ps, Clock_AT=%.1f ps, slack=%.1f ps",
                                worst_ep_pin, worst_AT_R, worst_AT_F, worst_ck_AT, slacks_flat_dbg.min().item())

                    logging.info("[Diag] worst endpoint pin_id=%d | AT_rise=%.1f ps, AT_fall=%.1f ps, slack=%.1f ps",
                                worst_ep_pin, worst_AT_R, worst_AT_F, slacks_flat_dbg.min().item())

                    # 4. position scale 확인 (wire delay near-zero 원인 진단)
                    logging.info("[Diag] pin_x range: [%.1f, %.1f] site, pin_y range: [%.1f, %.1f] site | pos_to_um=%.4f",
                                pin_pos_x.min().item(), pin_pos_x.max().item(),
                                pin_pos_y.min().item(), pin_pos_y.max().item(),
                                self.pos_to_um)
                    logging.info("[Diag] die expected in site units: x≈%.0f, y≈%.0f (8543µm / %.4fµm/site)",
                                8543.0 / self.pos_to_um, 3131.0 / self.pos_to_um, self.pos_to_um)

            PS_TO_NS = 1e-3
            logging.info("iter %d | WNS: %.4f ns, TNS: %.4f ns | WNS_penalty: %.4f ps, TNS_penalty: %.4f ps",
                        iteration, wns_val * PS_TO_NS, tns_val * 1e-5, wns_penalty.item(), tns_penalty.item())

            return timing_loss.to(pos.dtype)

        else:
            return torch.tensor(0.0, device=self.device, dtype=pos.dtype)