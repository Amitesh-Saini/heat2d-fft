// config_io.cpp
// Responsibility:
//   Implement JSON-based configuration input/output for simulation runs.
//
//   This file should read a JSON config file, validate the requested settings,
//   and convert them into the project's C++ configuration objects.
//
//   Expected configuration categories include:
//     - grid and domain settings,
//     - heat-equation parameters,
//     - output-time settings,
//     - initial-condition type and parameters,
//     - output-folder / writer settings,
//     - optional diagnostics or validation flags.
//
//   This module is the bridge between human-editable config files and the
//   strongly typed C++ objects used by the solver pipeline.

#include "config_io.hpp"         

#include "run_config.hpp"          
#include "time_grid.hpp"          
#include "initial_conditions.hpp"  
#include "fft1d.hpp"

#include <nlohmann/json.hpp>       

#include <fstream>               
#include <sstream>                 
#include <string>                 
#include <vector>                  
#include <stdexcept>  
#include <set>


std::string read_text_file(const std::string& path){

    std::ifstream file(path);
    
    if(!file.is_open()) throw std::runtime_error("read_text_file: File cannot be fetched" + path);

    std::stringstream file_text;
    file_text << file.rdbuf();

    return file_text.str();
}


void validate_run_config(const RunConfig& config){

    if (config.schema_version != current_schema_version)
        throw std::invalid_argument("validate_run_config: unsupported schema_version " + std::to_string(config.schema_version) + " (this build expects " + std::to_string(current_schema_version) + ")");

    if(!is_power_of_two(config.solver.nx) || !is_power_of_two(config.solver.ny)) throw std::invalid_argument("validate_run_config Grid size must be power of 2");

    if(config.output.output_path.empty()) throw std::invalid_argument("validate_run_config: Output path empty");

    if(config.output.gzip_level < 0 || config.output.gzip_level > 9) throw std::invalid_argument("validate_run_config: gzip level must be in [0, 9]");
    

    const bool is_fourier_ic = std::holds_alternative<SingleFourierModeIcParams>(config.initial_condition) || std::holds_alternative<MultiFourierModeIcParams>(config.initial_condition);

    if (config.diagnostics.compute_analytic_error && !is_fourier_ic){ 
        throw std::invalid_argument("validate_run_config: analytic error is only available for single- or " "multi-mode Fourier initial conditions");
        }
}


void reject_unknown_keys(const nlohmann::json& obj, const std::set<std::string>& allowed, const std::string& context){

    for(auto it = obj.begin(); it != obj.end(); ++it){
        if(!allowed.count(it.key()))
            throw std::invalid_argument(context + ": unknown key '" + it.key() + "'");
    }
}


RunConfig parse_run_config_json(const std::string& json_text){

    nlohmann::json data;

    try{

        data = nlohmann::json::parse(json_text);
    } catch(const nlohmann::json::parse_error& e){

        throw std::invalid_argument(std::string("parse_run_config_json: Invalid Json") + e.what());
    }

    const int schema = data.at("schema_version").get<int>();

    if(schema != current_schema_version){
        throw std::invalid_argument("parse_run_config_json: unsupported schema_version " + std::to_string(schema) + " (expected " + std::to_string(current_schema_version) + ")");
        
    }

    reject_unknown_keys(data, {"schema_version", "solver", "fft_backend",
    "initial_condition", "time", "output", "diagnostics"}, "top-level");


    RunConfig config;


    // Solver Field Check

    const nlohmann::json& solver_obj = data.at("solver");
    reject_unknown_keys(solver_obj, {"nx", "ny", "Lx", "Ly", "alpha"}, "solver");

    config.solver.nx = solver_obj.at("nx").get<std::size_t>();
    config.solver.ny = solver_obj.at("ny").get<std::size_t>();
    config.solver.Lx = solver_obj.at("Lx").get<Real>();
    config.solver.Ly = solver_obj.at("Ly").get<Real>();
    config.solver.alpha = solver_obj.at("alpha").get<Real>();


    // FFT Backend Field Check

    config.fft_backend = fft_backend_from_string(data.at("fft_backend").get<std::string>());


    // Initial Condition Field Check

    const nlohmann::json& ic_obj = data.at("initial_condition");
    const std::string ic_type = ic_obj.at("type").get<std::string>();

    if(ic_type == "gaussian"){

        reject_unknown_keys(ic_obj, {"type", "amplitude", "sigma", "image_radius_x", "image_radius_y"}, "gaussian IC");

        GaussianIcParams p;
        p.amplitude = ic_obj.at("amplitude").get<Real>();
        if(ic_obj.contains("sigma")) p.sigma = ic_obj.at("sigma").get<Real>();  
        p.image_radius_x = ic_obj.at("image_radius_x").get<std::size_t>();
        p.image_radius_y = ic_obj.at("image_radius_y").get<std::size_t>();
        config.initial_condition = p;      
    }

    else if (ic_type == "hot_square"){

        reject_unknown_keys(ic_obj, {"type", "amplitude", "width_x", "width_y", "smooth_width_x", "smooth_width_y"}, "hot_square IC");

        HotSquareIcParams p;
        p.amplitude = ic_obj.at("amplitude").get<Real>();   

        if (ic_obj.contains("width_x"))        p.width_x = ic_obj.at("width_x").get<Real>();
        if (ic_obj.contains("width_y"))        p.width_y = ic_obj.at("width_y").get<Real>();
        if (ic_obj.contains("smooth_width_x")) p.smooth_width_x = ic_obj.at("smooth_width_x").get<Real>();
        if (ic_obj.contains("smooth_width_y")) p.smooth_width_y = ic_obj.at("smooth_width_y").get<Real>();

        config.initial_condition = p;
    }

    else if(ic_type == "constant"){

        reject_unknown_keys(ic_obj, {"type", "T0"}, "constant IC");

        ConstantIcParams p;
        p.T0 = ic_obj.at("T0").get<Real>();
        config.initial_condition = p;
    }

    else if(ic_type == "single_fourier_mode"){

        reject_unknown_keys(ic_obj, {"type", "kx", "ky", "amplitude", "phase"}, "single_fourier_mode IC");

        SingleFourierModeIcParams p;
        p.amplitude = ic_obj.at("amplitude").get<Real>();
        p.kx = ic_obj.at("kx").get<std::ptrdiff_t>();
        p.ky = ic_obj.at("ky").get<std::ptrdiff_t>();
        p.phase = ic_obj.at("phase").get<Real>();
        config.initial_condition = p;
    }

   else if(ic_type == "multi_fourier_mode"){

    reject_unknown_keys(ic_obj, {"type", "modes"}, "multi_fourier_mode IC");

    MultiFourierModeIcParams p;

    if(ic_obj.contains("modes")){

        const nlohmann::json& modes_array = ic_obj.at("modes");

        if(!modes_array.is_array()){
            throw std::invalid_argument(
                "multi_fourier_mode IC: 'modes' must be an array");
        }

        for(const nlohmann::json& mode_obj : modes_array){

            reject_unknown_keys(mode_obj, {"kx", "ky", "amplitude", "phase"},
                                "multi_fourier_mode IC: mode entry");

            FourierMode2D mode;
            mode.kx        = mode_obj.at("kx").get<std::ptrdiff_t>();
            mode.ky        = mode_obj.at("ky").get<std::ptrdiff_t>();
            mode.amplitude = mode_obj.at("amplitude").get<Real>();
            mode.phase     = mode_obj.at("phase").get<Real>();

            p.modes.push_back(mode);
        }
    }

    config.initial_condition = p;

    }

    else {
        throw std::invalid_argument("initial_condition: unknown type '" + ic_type + "'");
    }



    // Time Field Check

    const nlohmann::json& time_obj = data.at("time");
    const std::string time_type = time_obj.at("mode").get<std::string>();

    if(time_type == "uniform"){

        reject_unknown_keys(time_obj, {"mode", "t_start", "t_end", "num_snapshots"}, "uniform time");

        UniformTimeSpec spec;
        spec.t_start = time_obj.at("t_start").get<Real>();
        spec.t_end = time_obj.at("t_end").get<Real>();
        spec.num_snapshots = time_obj.at("num_snapshots").get<std::size_t>();
        config.time_spec = spec;
    }
    else if(time_type == "explicit"){

        reject_unknown_keys(time_obj, {"mode", "times"}, "explicit time");

        const nlohmann::json& times_array = time_obj.at("times");
        if(!times_array.is_array())
            throw std::invalid_argument("explicit time: 'times' must be an array");

        ExplicitTimeSpec spec;
        spec.times = times_array.get<RealVec>();
        config.time_spec = spec;
    }

    else{
        throw std::invalid_argument("time: unknown mode '" + time_type + "'");
    }


    // Output Field Check

    const nlohmann::json& output_obj = data.at("output");

    reject_unknown_keys(output_obj, {"output_path", "overwrite", "gzip_level"}, "output");

    OutputSettings output;

    output.output_path = output_obj.at("output_path").get<std::string>();

    if (output_obj.contains("overwrite"))
        output.overwrite = output_obj.at("overwrite").get<bool>();

    if (output_obj.contains("gzip_level"))
        output.gzip_level = output_obj.at("gzip_level").get<int>();

    config.output = output;



    // Diagnostics Field Check

    const nlohmann::json& diagnostics_obj = data.at("diagnostics");

    reject_unknown_keys(diagnostics_obj, {"enabled", "compute_analytic_error"}, "diagnostics");

    DiagnosticsSettings diagnostics;

    if (diagnostics_obj.contains("enabled"))
        diagnostics.enabled = diagnostics_obj.at("enabled").get<bool>();

    if (diagnostics_obj.contains("compute_analytic_error"))
        diagnostics.compute_analytic_error = diagnostics_obj.at("compute_analytic_error").get<bool>();

    config.diagnostics = diagnostics;


    // Schema and Source File

    config.schema_version = schema;
    config.source_json_text = json_text;

    validate_run_config(config);

    return config;
}



RunConfig load_run_config_from_json(const std::string& path){
    return parse_run_config_json(read_text_file(path));
}