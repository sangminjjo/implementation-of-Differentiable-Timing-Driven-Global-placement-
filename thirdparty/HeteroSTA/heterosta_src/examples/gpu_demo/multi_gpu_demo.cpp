#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>
#include <heterosta.h>

// --- Thread-Safe Logging Setup ---

// Thread-local storage pointers for directing log output.
thread_local std::ostream* g_thread_log_out = nullptr;
thread_local std::ostream* g_thread_log_err = nullptr;

// C-style callback function to route Rust logger messages.
extern "C" void cpp_log_callback(uint8_t level, const char* message) {
    const char* level_str;
    switch (level) {
        case 1: level_str = "ERROR"; break;
        case 2: level_str = "WARN "; break;
        case 3: level_str = "INFO "; break;
        case 4: level_str = "DEBUG"; break;
        case 5: level_str = "TRACE"; break;
        default: level_str = "UNKNW"; break;
    }
    
    // Route logs based on severity level.
    if (level <= 2) {
        if (g_thread_log_err) {
            *g_thread_log_err << "[" << level_str << "] " << message << std::endl;
        } else {
            std::cerr << "[" << level_str << "] " << message << std::endl;
        }
    } else {
        if (g_thread_log_out) {
            *g_thread_log_out << "[" << level_str << "] " << message << std::endl;
        } else {
            std::cout << "[" << level_str << "] " << message << std::endl;
        }
    }
}

// A hardcoded fallback license..
const char* hardcode_lic = nullptr;

// run the entire STA flow on a single GPU.
void run_sta_on_gpu(int gpu_id) {
    // 1. Setup thread-specific file logging.
    std::string log_filename = "results/multi_gpu_demo/gpu_" + std::to_string(gpu_id) + ".log";
    std::ofstream log_file(log_filename);

    // Point the thread-local streams to this thread's log file.
    g_thread_log_out = &log_file;
    g_thread_log_err = &log_file;
    
    // --- Configuration ---
    log_file << "--- Log for GPU " << gpu_id << " ---" << std::endl;
    const char* netlist_verilog = "../common/data/simple.v";
    const char* liberty_min = "../common/data/simple_Early.lib";
    const char* liberty_max = "../common/data/simple_Late.lib";
    const char* spef_file = "../common/data/simple.spef";
    const char* sdc_file = "../common/data/simple.sdc";
    
    // Generate unique output paths for this thread.
    std::string sdf_out_path = "results/multi_gpu_demo/output_gpu_" + std::to_string(gpu_id) + ".sdf";
    std::string rpt_out_path = "results/multi_gpu_demo/output_gpu_" + std::to_string(gpu_id) + ".rpt";
    
    // 2. Create a new STA instance on the assigned GPU.
    STAHoldings* sta = heterosta_new_with_device(gpu_id);
    if (!sta) {
        log_file << "Error: Failed to create STAHoldings instance." << std::endl;
        return;
    }

    // 3. Select a delay calculator based on the GPU ID to demonstrate flexibility.
    switch (gpu_id % 3) {
        case 0:
            log_file << "Using Elmore delay calculator." << std::endl;
            heterosta_set_delay_calculator_elmore(sta);
            break;
        case 1:
            log_file << "Using Scaled Elmore delay calculator." << std::endl;
            heterosta_set_delay_calculator_elmore_scaled(sta);
            break;
        case 2:
            log_file << "Using Arnoldi delay calculator." << std::endl;
            heterosta_set_delay_calculator_arnoldi(sta);
            break;
    }

    // --- Main STA Flow ---
    if (!heterosta_read_liberty(sta, EARLY, liberty_min) || !heterosta_read_liberty(sta, LATE, liberty_max)) {
        log_file << "Error on Reading Liberty." << std::endl; heterosta_free(sta); return;
    }
    if (!heterosta_read_netlist(sta, netlist_verilog, nullptr)) {
        log_file << "Error on Reading Netlist." << std::endl; heterosta_free(sta); return;
    }
    if (!heterosta_read_spef(sta, spef_file)) {
        log_file << "Error on Reading SPEF." << std::endl; heterosta_free(sta); return;
    }

    heterosta_flatten_all(sta);
    heterosta_build_graph(sta);
    heterosta_zero_slew(sta);
    
    if (!heterosta_read_sdc(sta, sdc_file)) {
        log_file << "Error on Reading SDC." << std::endl; heterosta_free(sta); return;
    }
    
    const bool use_cuda = true;
    heterosta_update_delay(sta, use_cuda);
    heterosta_update_arrivals(sta, use_cuda);

    // Report timing analysis results.
    float wns = 0.0, tns = 0.0;
    if (heterosta_report_wns_tns_max(sta, &wns, &tns, use_cuda)) {
         log_file << "\n--- Timing Results ---\n"
                  << "WNS (setup): " << wns << "\n"
                  << "TNS (setup): " << tns << std::endl;
    } else {
        log_file << "Warning: Could not report WNS/TNS." << std::endl;
    }
    
    if (!heterosta_report_delay_sdf(sta, sdf_out_path.c_str())) {
        log_file << "Warning: Failed to write SDF file." << std::endl;
    }

    heterosta_dump_paths_max_to_file(sta, 100, 100, rpt_out_path.c_str(), use_cuda);

    // Free resources for this STA instance.
    heterosta_free(sta);
    
    log_file << "\nSTA on GPU " << gpu_id << " finished successfully." << std::endl;

    // Reset thread-local pointers as a good practice before the thread exits.
    g_thread_log_out = nullptr;
    g_thread_log_err = nullptr;

    log_file.close();
}


int main(int argc, char* argv[]) {
    const std::string output_dir = "results/multi_gpu_demo";
    try {
        std::filesystem::create_directories(output_dir);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "[FATAL ERROR] Could not create output directory '" << output_dir << "': " << e.what() << std::endl;
        return 1;
    }
    
    // 1. Initialize license, prioritizing environment variable over hardcoded fallback.
    const char* lic = std::getenv("HeteroSTA_Lic");
    if (lic == nullptr) {
        if (hardcode_lic == nullptr) {
            std::cerr << "[FATAL ERROR] License not found." << std::endl;
            std::cerr << "               Please either set the 'HeteroSTA_Lic' environment variable" << std::endl;
            std::cerr << "               or provide a hardcoded license in the source code." << std::endl;
            return 1;
        }
        lic = hardcode_lic;
        std::cout << "[INFO] 'HeteroSTA_Lic' environment variable not found. Using hardcoded license." << std::endl;
    } else {
        std::cout << "[INFO] Successfully loaded license from 'HeteroSTA_Lic' environment variable." << std::endl;
    }

    // 2. Initialize global logger and validate the license.
    heterosta_init_logger(cpp_log_callback);
    heterosta_init_license(lic);

    // 3. Query the number of available GPUs.
    size_t num_gpus = heterosta_get_num_cuda_devices();
    if (num_gpus == 0) {
        std::cerr << "No CUDA devices found. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Found " << num_gpus << " available CUDA devices." << std::endl;
    // Limit the demo to 3 GPUs for brevity, but you can remove this line to use all.
    num_gpus = num_gpus <= 3 ? num_gpus : 3;
    std::cout << "Starting parallel STA on " << num_gpus << " GPUs... Each thread will log to 'results/multi_gpu_demo/gpu_N.log'." << std::endl;

    // 4. Create and launch a thread for each GPU.
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_gpus; ++i) {
        threads.emplace_back(run_sta_on_gpu, i);
    }

    // 5. Wait for all threads to complete and report progress.
    std::cout << "\nWaiting for all STA tasks to finish..." << std::endl;
    size_t finished_count = 0;
    for (auto& th : threads) {
        th.join(); // Block until this thread has finished.
        std::cout << ">>> Progress: Task for GPU " << finished_count << " has completed." << std::endl;
        finished_count++;
    }

    std::cout << "\nAll parallel STA tasks have completed. Check log files for details." << std::endl;

    return 0;
}