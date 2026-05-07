#pragma once
#include <heterosta.h>
#include <netlistdb.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>

using pin_direction_type_t = uint8_t;
enum : pin_direction_type_t {
    kPinDirectReceiver = 0b01,
    kPinDirectDriver = 0b10,
    kPinDirectBidirect = kPinDirectDriver | kPinDirectReceiver,
};

template<typename T>
std::vector<const char*> make_c_str_view(const std::vector<T>& vec) {
    std::vector<const char*> view;
    view.reserve(vec.size());
    for (const auto& item : vec) {
        view.push_back(item.c_str());
    }
    return view;
}

namespace db {

struct Pin;
struct Net;
struct Cell;

struct Pin {
    std::string name;
    pin_direction_type_t direct;
    Cell* cell = nullptr;
    Net*  net  = nullptr;
};

struct Net {
    std::string name;
    std::vector<Pin*> pins;
};

struct Cell {
    std::string name;
    std::string type;
    std::vector<Pin*> pins;
};

class NormalPlaceDB {
public:
    std::string top_design_name;
    std::vector<std::unique_ptr<Cell>> cells;
    std::vector<std::unique_ptr<Net>>  nets;
    std::vector<std::unique_ptr<Pin>>  pins;

    NormalPlaceDB() = default;

    Cell* create_cell(const std::string& name, const std::string& type);
    Pin*  create_pin(Cell* parent_cell, const std::string& name, pin_direction_type_t dir);
    Net*  create_net(const std::string& name);
    void  connect_pin_to_net(Pin* pin, Net* net);

    bool read_verilog(const std::string& file_path);

private:
    std::string trim(const std::string& str);
    std::vector<std::string> split(const std::string& s, char delimiter);
};

/// flat version of databbase, for demo purposes
class FlatPlaceDB {
public:
    using index_type = int32_t;
    std::string top_design_name;

    // 1. Cells/Nodes
    std::vector<std::string> node_names; ///< cell instance names
    std::vector<std::string> node_types; ///< cell type names

    // 2. Pins
    std::vector<std::string> pin_names;           ///< pin names
    std::vector<pin_direction_type_t> pin_direct; ///< pin directions
    size_t pin2cell_top_level_count{};
    index_type num_virtual_hierarchy_pin = 0;   ///< number of virtual hierarchy pin

    // 3. Nets
    std::vector<std::string> net_names;

    // 4. Connectivity Maps
    std::vector<uintptr_t> pin2node_map; ///< pin -> node map
    std::vector<uintptr_t> pin2net_map;  ///< pin -> net map

    // 5. Power/Ground Grid
    std::vector<index_type> pin_zero_indices; ///< id of pins connected to ground
    std::vector<index_type> pin_one_indices;  ///< id of pins connected to power


    FlatPlaceDB() = default;
    FlatPlaceDB(const NormalPlaceDB& normal_db);

    NetlistDB* build_netlistdb(bool verbose=false);
};

} // namespace db