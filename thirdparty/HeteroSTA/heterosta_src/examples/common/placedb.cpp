// flat_placedb.cpp
#include <placedb.h>
#include <iomanip>

namespace db {

Cell* NormalPlaceDB::create_cell(const std::string& name, const std::string& type) {
    auto new_cell_ptr = std::make_unique<Cell>();

    new_cell_ptr->name = name;
    new_cell_ptr->type = type;

    Cell* raw_ptr = new_cell_ptr.get();

    cells.push_back(std::move(new_cell_ptr));

    return raw_ptr;
}

Pin* NormalPlaceDB::create_pin(Cell* parent_cell, const std::string& name, pin_direction_type_t dir) {
    auto new_pin_ptr = std::make_unique<Pin>();

    new_pin_ptr->name = name;
    new_pin_ptr->direct = dir;
    new_pin_ptr->cell = parent_cell;

    Pin* raw_ptr = new_pin_ptr.get();

    // If it's an instance pin, add it to the cell's pin list and create a hierarchical name.
    if (parent_cell != nullptr) {
        parent_cell->pins.push_back(raw_ptr);
        new_pin_ptr->name = parent_cell->name + "/" + name; // e.g., "u1/A"
    }

    pins.push_back(std::move(new_pin_ptr));

    return raw_ptr;
}

Net* NormalPlaceDB::create_net(const std::string& name) {
    auto new_net_ptr = std::make_unique<Net>();

    new_net_ptr->name = name;

    Net* raw_ptr = new_net_ptr.get();

    nets.push_back(std::move(new_net_ptr));

    return raw_ptr;
}

void NormalPlaceDB::connect_pin_to_net(Pin* pin, Net* net) {
    assert(pin != nullptr && "Pin cannot be null.");
    assert(net != nullptr && "Net cannot be null.");

    pin->net = net;

    auto it = std::find(net->pins.begin(), net->pins.end(), pin);
    if (it == net->pins.end()) {
        net->pins.push_back(pin);
    }
}

std::string NormalPlaceDB::trim(const std::string& str) {
    const std::string whitespace = " \t\r\n;";
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos) return "";
    const auto strEnd = str.find_last_not_of(whitespace);
    return str.substr(strBegin, strEnd - strBegin + 1);
}


std::vector<std::string> NormalPlaceDB::split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

bool NormalPlaceDB::read_verilog(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open Verilog file: " << file_path << std::endl;
        return false;
    }

    cells.clear();
    nets.clear();
    pins.clear();

    enum class ParseState { IDLE, IN_PORTS_DECL, IN_WIRES_DECL, IN_CELLS_INST };
    ParseState state = ParseState::IDLE;

    std::unordered_map<std::string, Net*> net_map;

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line.rfind("//", 0) == 0) continue;

        if (line.rfind("module", 0) == 0) {
            auto parts = split(line, ' ');
            if (parts.size() > 1) this->top_design_name = parts[1];
            continue;
        }
        if (line.rfind("input", 0) == 0 || line.rfind("output", 0) == 0) {
            state = ParseState::IN_PORTS_DECL;
        } else if (line.rfind("wire", 0) == 0) {
            state = ParseState::IN_WIRES_DECL;
        } else if (line.find("(") != std::string::npos && line.find(")") != std::string::npos && line.find(".") != std::string::npos) {
            state = ParseState::IN_CELLS_INST;
        } else if (line.rfind("endmodule", 0) == 0) {
            break;
        }

        switch (state) {
            case ParseState::IN_PORTS_DECL: {
                auto parts = split(line, ' ');
                if (parts.size() > 1) {
                    std::string port_name = trim(parts[1]);
                    if (net_map.find(port_name) == net_map.end()) {
                        Net* new_net = this->create_net(port_name);
                        net_map[port_name] = new_net;
                        pin_direction_type_t dir;
                        if (line.rfind("input", 0) == 0) {
                            dir = kPinDirectDriver;
                        } else { // line must be "output ..."
                            dir = kPinDirectReceiver;
                        }
                        Pin* new_pin = this->create_pin(nullptr, port_name, dir);
                        this->connect_pin_to_net(new_pin, new_net);
                    }
                }
                break;
            }
            case ParseState::IN_WIRES_DECL: {
                auto parts = split(line, ' ');
                if (parts.size() > 1) {
                    std::string net_name = trim(parts[1]);
                    if (net_map.find(net_name) == net_map.end()) {
                        Net* new_net = this->create_net(net_name);
                        net_map[net_name] = new_net;
                    }
                }
                break;
            }
            case ParseState::IN_CELLS_INST: {
                std::stringstream ss(line);
                std::string cell_type, inst_name;
                ss >> cell_type >> inst_name;

                Cell* new_cell = this->create_cell(inst_name, cell_type);

                size_t start_pos = line.find('(') + 1;
                size_t end_pos = line.rfind(')');
                std::string connections_str = line.substr(start_pos, end_pos - start_pos);
                
                auto pin_connections = split(connections_str, ',');
                for (const auto& conn : pin_connections) {
                    std::string clean_conn = trim(conn);
                    size_t pin_name_start = clean_conn.find('.') + 1;
                    size_t pin_name_end = clean_conn.find('(');
                    std::string pin_name = clean_conn.substr(pin_name_start, pin_name_end - pin_name_start);
                    
                    size_t net_name_start = pin_name_end + 1;
                    size_t net_name_end = clean_conn.rfind(')');
                    std::string net_name = clean_conn.substr(net_name_start, net_name_end - net_name_start);

                    Net* target_net = nullptr;
                    if (net_map.count(net_name)) {
                        target_net = net_map.at(net_name);
                    } else {
                        target_net = this->create_net(net_name);
                        net_map[net_name] = target_net;
                    }

                    pin_direction_type_t dir;
                    if (pin_name == "a" || pin_name == "b" || pin_name == "d" || pin_name == "ck") {
                        dir = kPinDirectReceiver;
                    } else if (pin_name == "o" || pin_name == "q") {
                        dir = kPinDirectDriver;
                    } else {
                        dir = kPinDirectBidirect;
                    }

                    Pin* new_pin = this->create_pin(new_cell, pin_name, dir);
                    this->connect_pin_to_net(new_pin, target_net);
                }
                break;
            }
            default:
                break;
        }
    }

    return true;
}

FlatPlaceDB::FlatPlaceDB(const NormalPlaceDB& normal_db) {
    this->top_design_name = normal_db.top_design_name;

    std::unordered_map<const Cell*, uintptr_t> cell_ptr_to_id;
    std::unordered_map<const Net*, uintptr_t> net_ptr_to_id;
    std::unordered_map<const Pin*, uintptr_t> pin_ptr_to_id;

    this->node_names.reserve(normal_db.cells.size());
    this->node_types.reserve(normal_db.cells.size());
    this->pin_names.reserve(normal_db.pins.size());
    this->pin_direct.reserve(normal_db.pins.size());
    this->net_names.reserve(normal_db.nets.size());
    this->pin2node_map.resize(normal_db.pins.size());
    this->pin2net_map.resize(normal_db.pins.size());

    uintptr_t current_node_id = 0;
    for (const auto& cell_ptr : normal_db.cells) {
        this->node_names.push_back(cell_ptr->name);
        this->node_types.push_back(cell_ptr->type);
        cell_ptr_to_id[cell_ptr.get()] = current_node_id++;
    }

    uintptr_t current_net_id = 0;
    for (const auto& net_ptr : normal_db.nets) {
        this->net_names.push_back(net_ptr->name);
        net_ptr_to_id[net_ptr.get()] = current_net_id++;
    }

    // Maintain a specific order: instance pins first, then top-level ports.
    // This is crucial for the logic in build_netlistdb.
    uintptr_t current_pin_id = 0;
    for (const auto& cell_ptr : normal_db.cells) {
        for (const Pin* pin_obj_ptr : cell_ptr->pins) {
            this->pin_names.push_back(pin_obj_ptr->name);
            this->pin_direct.push_back(pin_obj_ptr->direct);
            pin_ptr_to_id[pin_obj_ptr] = current_pin_id++;
        }
    }
    this->pin2cell_top_level_count = 0;
    for (const auto& pin_ptr : normal_db.pins) {
        if (pin_ptr->cell == nullptr) {
            this->pin_names.push_back(pin_ptr->name);
            this->pin_direct.push_back(pin_ptr->direct);
            pin_ptr_to_id[pin_ptr.get()] = current_pin_id++;
            this->pin2cell_top_level_count++;
        }
    }

    for (const auto& pin_ptr : normal_db.pins) {
        uintptr_t pin_id = pin_ptr_to_id.at(pin_ptr.get());

        if (pin_ptr->net != nullptr) {
            uintptr_t net_id = net_ptr_to_id.at(pin_ptr->net);
            this->pin2net_map[pin_id] = net_id;
        }

        if (pin_ptr->cell != nullptr) {
            uintptr_t node_id = cell_ptr_to_id.at(pin_ptr->cell);
            this->pin2node_map[pin_id] = node_id;
        }
    }
    
    for(const auto& net_ptr : normal_db.nets) {
        if (net_ptr->name == "VSS" || net_ptr->name == "GND" || net_ptr->name == "1'b0") {
            for (const Pin* pin_obj_ptr : net_ptr->pins) {
                uintptr_t pin_id = pin_ptr_to_id.at(pin_obj_ptr);
                this->pin_zero_indices.push_back(pin_id);
            }
        } else if (net_ptr->name == "VDD" || net_ptr->name == "VCC" || net_ptr->name == "1'b1") {
            for (const Pin* pin_obj_ptr : net_ptr->pins) {
                uintptr_t pin_id = pin_ptr_to_id.at(pin_obj_ptr);
                this->pin_one_indices.push_back(pin_id);
            }
        }
    }
}

NetlistDB* FlatPlaceDB::build_netlistdb(bool verbose) {
    NetlistDBCppInterface netlistDbCppInterface{};

    const size_t num_inst = node_names.size();
    
    // 1. Build cell database
    std::vector<const char*> cellnames_view;
    std::vector<const char*> celltypes_view;

    auto cellnames_c_str = make_c_str_view(node_names);
    auto celltypes_c_str = make_c_str_view(node_types);

    assert(cellnames_c_str.size() == num_inst);
    assert(celltypes_c_str.size() == num_inst);

    cellnames_view.reserve(num_inst + 1);
    celltypes_view.reserve(num_inst + 1);
    
    // Cell 0: top design
    cellnames_view.push_back("");
    celltypes_view.push_back(top_design_name.c_str());

    for (const auto &cellname : cellnames_c_str) {
        cellnames_view.push_back(cellname);
    }
    for (const auto &celltype : celltypes_c_str) {
        celltypes_view.push_back(celltype);
    }

    netlistDbCppInterface.top_design_name = top_design_name.c_str();
    netlistDbCppInterface.num_cells = num_inst + 1;
    netlistDbCppInterface.cellname_array = cellnames_view.data();
    netlistDbCppInterface.celltype_array = celltypes_view.data();

    // 2. Build pin database
    std::vector<const char*> pin_names_view;
    std::vector<uint8_t> pin_dir_arr;
    std::vector<uintptr_t> pin2node_map_view;
    std::vector<uintptr_t> pin2net_map_view;
    std::vector<const char*> net_names_view;

    pin_names_view.reserve(pin_names.size());
    pin_dir_arr.reserve(pin_direct.size());
    pin2node_map_view.reserve(pin2node_map.size());
    pin2net_map_view.reserve(pin2net_map.size());
    net_names_view.reserve(net_names.size());

    pin_names_view.reserve(pin_names.size());
    for (const auto& name : pin_names) {
        pin_names_view.push_back(name.c_str());
    }

    pin_dir_arr.reserve(pin_direct.size());
    for (auto dir : pin_direct) {
        if (dir == kPinDirectReceiver) {
            pin_dir_arr.push_back(NetlistDirection::I);
        } else if (dir == kPinDirectDriver) {
            pin_dir_arr.push_back(NetlistDirection::O);
        } else {
            pin_dir_arr.push_back(NetlistDirection::Unknown);
        }
    }
    
    for (auto inst_index: pin2node_map) {
        pin2node_map_view.push_back(inst_index + 1);
    }
    std::fill(pin2node_map_view.rbegin() + num_virtual_hierarchy_pin, 
        pin2node_map_view.rbegin() + pin2cell_top_level_count + num_virtual_hierarchy_pin, 0);
    std::copy(pin2net_map.begin(), pin2net_map.end(), std::back_inserter(pin2net_map_view));
    for (const auto& name : net_names) {
        net_names_view.push_back(name.c_str());
    }

    netlistDbCppInterface.num_pins = pin_names_view.size();
    netlistDbCppInterface.num_ports = pin2cell_top_level_count + num_virtual_hierarchy_pin;
    netlistDbCppInterface.pinname_array = pin_names_view.data();
    netlistDbCppInterface.pindirection_array = pin_dir_arr.data();

    // 3.1  Set the insts pins
    netlistDbCppInterface.num_nets = net_names_view.size();
    netlistDbCppInterface.netname_array = net_names_view.data();
    netlistDbCppInterface.pin2cell_array = pin2node_map_view.data();
    netlistDbCppInterface.pin2net_array = pin2net_map_view.data();

    std::vector<uintptr_t> nets_zero_array;
    std::vector<uintptr_t> nets_one_array;
    nets_zero_array.reserve(pin_zero_indices.size());
    nets_one_array.reserve(pin_one_indices.size());
    for (auto index : pin_zero_indices) {
        nets_zero_array.push_back(pin2net_map[index]);
    }
    for (auto index : pin_one_indices) {
        nets_one_array.push_back(pin2net_map[index]);
    }
    netlistDbCppInterface.num_nets_zero = nets_zero_array.size();
    netlistDbCppInterface.num_nets_one = nets_one_array.size();
    netlistDbCppInterface.nets_zero_array = nets_zero_array.data();
    netlistDbCppInterface.nets_one_array = nets_one_array.data();

    if (verbose) {
        // --- Preamble and Summary ---
        // Print a main header and a summary of the database contents for a quick overview.
        // This includes key counts like the number of cells, pins, and top-level ports.
        std::cout << "\n## NetlistDB Cpp Interface Data Mapping Explanation\n\n"
                << "This document explains the mapping from original database names to the integer indices\n"
                << "used in the arrays passed to the `NetlistDB` C-style interface.\n\n"
                << "### Summary of Database Content\n\n"
                << "- Top Design Name: " << netlistDbCppInterface.top_design_name << "\n"
                << "- Total Cells (Instances + Top): " << netlistDbCppInterface.num_cells << "\n"
                << "- Total Pins (Instance Pins + Ports): " << netlistDbCppInterface.num_pins << "\n"
                << "- Total Top-Level Ports: " << netlistDbCppInterface.num_ports << "\n"
                << "- Total Nets: " << netlistDbCppInterface.num_nets << "\n\n";

        // --- Section 1: Cell Arrays ---
        // Explain the mapping for cellname_array and celltype_array.
        // Cell index 0 is always reserved for the top-level design.
        std::cout << "### 1. Cell/Instance Arrays\n\n"
                << "The `cellname_array` and `celltype_array` map a cell index to its instance name and type.\n"
                << "**Note:** `Cell[0]` is reserved for the top-level design module.\n\n"
                // Adjusted headers to match the user's request.
                << "| Cell Index | `cellname_array` | `celltype_array` |\n"
                << "|:----------:|:----------------:|:----------------:|\n";
        for (size_t i = 0; i < netlistDbCppInterface.num_cells; ++i) {
            // Adjusted setw for new, shorter headers while keeping content aligned.
            std::cout << "| " << std::setw(10) << i << " | "
                    << std::setw(16) << netlistDbCppInterface.cellname_array[i] << " | "
                    << std::setw(16) << netlistDbCppInterface.celltype_array[i] << " |\n";
        }
        std::cout << "\n";

        // --- Section 2: Net Array ---
        // Explain the mapping for the netname_array.
        std::cout << "### 2. Net Array\n\n"
                << "The `netname_array` maps a net index to its original name in the design.\n\n"
                // Adjusted header.
                << "| Net Index  | `netname_array` |\n"
                << "|:----------:|:---------------:|\n";
        for (size_t i = 0; i < netlistDbCppInterface.num_nets; ++i) {
            // Adjusted setw for alignment.
            std::cout << "| " << std::setw(10) << i << " | "
                    << std::setw(15) << netlistDbCppInterface.netname_array[i] << " |\n";
        }
        std::cout << "\n";

        // --- Section 3: Pin and Connectivity Arrays ---
        // This is the main table, showing the complete mapping for each pin.
        std::cout << "### 3. Pin and Connectivity Arrays\n\n"
                << "This table shows the comprehensive mapping for each pin, including its name, direction, owner cell, and connected net.\n\n"
                // Adjusted headers to match the user's request.
                << "| Pin Index | `pinname_array` | `pindirection_array`      | `pin2cell_array` | Cell Name | `pin2net_array` |   Net Name   |\n"
                << "|:---------:|:---------------:|:-------------------------:|:----------------:|:---------:|:---------------:|:-------------|\n";
        for (size_t i = 0; i < netlistDbCppInterface.num_pins; ++i) {
            // Convert the pin direction enum to the requested string format.
            uint8_t dir_val = netlistDbCppInterface.pindirection_array[i];
            std::string dir_str;
            switch (dir_val) {
                case NetlistDirection::I:       dir_str = "`NetlistDirection::I`";       break;
                case NetlistDirection::O:       dir_str = "`NetlistDirection::O`";       break;
                default:                        dir_str = "`NetlistDirection::Unknown`"; break;
            }

            // Look up cell and net names for context.
            uintptr_t cell_idx = netlistDbCppInterface.pin2cell_array[i];
            const char* cell_name = (cell_idx < netlistDbCppInterface.num_cells)
                                        ? netlistDbCppInterface.cellname_array[cell_idx] : "Error";
            uintptr_t net_idx = netlistDbCppInterface.pin2net_array[i];
            const char* net_name = (net_idx < netlistDbCppInterface.num_nets)
                                    ? netlistDbCppInterface.netname_array[net_idx] : "Error";

            // Completely recalibrated all setw values for optimal alignment.
            std::cout << "| " << std::left << std::setw(9)  << i << " | "
                    << std::setw(15) << netlistDbCppInterface.pinname_array[i] << " | "
                    << std::setw(25) << dir_str << " | "
                    << std::setw(16) << cell_idx << " | "
                    << std::setw(9) << cell_name << " | "
                    << std::setw(15) << net_idx << " | "
                    << std::setw(12) << net_name << " |\n";
        }
        // Return to default right alignment for subsequent tables.
        std::cout << std::right << "\n";

        // --- Section 4: Special Net Arrays (Power/Ground) ---
        // Explain the contents of the constant power (1) and ground (0) net arrays.
        std::cout << "### 4. Special Net Arrays (Power/Ground)\n\n"
                << "These arrays list the **indices of nets** connected to constant power (1) or ground (0).\n\n";

        // Power Nets Table (`nets_one_array`)
        std::cout << "**`nets_one_array` (Power Nets):**\n\n";
        if (netlistDbCppInterface.num_nets_one == 0) {
            std::cout << "This array is empty.\n\n";
        } else {
            // Adjusted headers and widths for clarity and consistency.
            std::cout << "| Index | `nets_one_array` | Net Name |\n"
                    << "|:-----:|:----------------:|:---------|\n";
            for (size_t i = 0; i < netlistDbCppInterface.num_nets_one; ++i) {
                uintptr_t net_idx = netlistDbCppInterface.nets_one_array[i];
                const char* net_name = (net_idx < netlistDbCppInterface.num_nets)
                                        ? netlistDbCppInterface.netname_array[net_idx] : "Error";
                std::cout << "| " << std::setw(5)  << i << " | "
                        << std::setw(16) << net_idx << " | "
                        << std::setw(8) << net_name << " |\n";
            }
            std::cout << "\n";
        }

        // Ground Nets Table (`nets_zero_array`)
        std::cout << "**`nets_zero_array` (Ground Nets):**\n\n";
        if (netlistDbCppInterface.num_nets_zero == 0) {
            std::cout << "This array is empty.\n\n";
        } else {
            // Adjusted headers and widths for clarity and consistency.
            std::cout << "| Index | `nets_zero_array` | Net Name |\n"
                    << "|:-----:|:-----------------:|:---------|\n";
            for (size_t i = 0; i < netlistDbCppInterface.num_nets_zero; ++i) {
                uintptr_t net_idx = netlistDbCppInterface.nets_zero_array[i];
                const char* net_name = (net_idx < netlistDbCppInterface.num_nets)
                                        ? netlistDbCppInterface.netname_array[net_idx] : "Error";
                std::cout << "| " << std::setw(5)  << i << " | "
                        << std::setw(17) << net_idx << " | "
                        << std::setw(8) << net_name << " |\n";
            }
            std::cout << "\n";
        }
        
        // Ensure all output is written to the console immediately.
        std::cout << std::flush;
    }
    
    return netlistdb_new(&netlistDbCppInterface);
}

} // namespace db