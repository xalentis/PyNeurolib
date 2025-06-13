#include "neuron_simulator.h"
#include "neuron_types.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <iomanip>

NeuronSimulator::NeuronSimulator() {
    for (int i = 0; i < NEURON_COUNT; ++i) {
        neurons[i] = nullptr;
    }
    sim_data.total_timesteps = 0;
    sim_data.total_spikes = 0;
}

NeuronSimulator::~NeuronSimulator() {
    cleanup_neurons();
}

void NeuronSimulator::initialize_neurons() {
    cleanup_neurons();
    neurons[0] = new PyramidalNeuron();
    neurons[1] = new PyramidalNeuron();
    neurons[2] = new PyramidalNeuron();
    neurons[3] = new PyramidalNeuron();
    neurons[4] = new Interneuron();
    neurons[5] = new PurkinjeNeuron();
    neurons[6] = new MotorNeuron();
    neurons[7] = new MotorNeuron();
    neurons[8] = new SensoryNeuron();
    neurons[9] = new SensoryNeuron();
}

void NeuronSimulator::cleanup_neurons() {
    for (int i = 0; i < NEURON_COUNT; ++i) {
        delete neurons[i];
        neurons[i] = nullptr;
    }
}

void NeuronSimulator::create_random_connections(int connection_density) {
    for (int i = 0; i < NEURON_COUNT; ++i) {
        for (int j = 0; j < connection_density; ++j) {
            int target = rand() % NEURON_COUNT;
            if (target != i && neurons[target]->get_dendrite_count() > 0) {
                int target_dendrite = rand() % neurons[target]->get_dendrite_count();
                float weight = 1.5f + static_cast<float>(rand()) / RAND_MAX * 3.0f;
                bool inhibitory = !neurons[i]->get_is_excitatory() || (rand() % 8 == 0);
                neurons[i]->connect_to_neuron(neurons[target], target_dendrite, weight, inhibitory);
            }
        }
    }
}

void NeuronSimulator::collect_membrane_data() {
    std::vector<float> potentials;
    float total_potential = 0.0f;
    
    for (int i = 0; i < NEURON_COUNT; ++i) {
        float potential = neurons[i]->get_membrane_potential();
        potentials.push_back(potential);
        total_potential += potential;
    }
    
    sim_data.membrane_potentials.push_back(potentials);
    sim_data.network_activity.push_back(total_potential / NEURON_COUNT);
}

void NeuronSimulator::record_spike_event(int timestep, int neuron_id) {
    sim_data.spike_events.push_back(std::make_pair(timestep, neuron_id));
}

void NeuronSimulator::apply_background_activity(float noise_probability) {
    for (int i = 0; i < NEURON_COUNT; ++i) {
        if (static_cast<float>(rand()) / RAND_MAX < noise_probability) {
            if (static_cast<float>(rand()) / RAND_MAX < 0.25f) {
                neurons[i]->spike();
            }
        }
    }
}

void NeuronSimulator::run_standard_simulation(int max_timesteps) {
    // clear previous data
    sim_data.membrane_potentials.clear();
    sim_data.spike_events.clear();
    sim_data.network_activity.clear();
    
    initialize_neurons();
    create_random_connections();
    
    int timestep = 0;
    int total_spikes = 0;
    
    while (timestep < max_timesteps) {
        collect_membrane_data();
        
        if (timestep % 2 == 0) {
            int stimulated_neuron = rand() % NEURON_COUNT;
            neurons[stimulated_neuron]->spike();
        }
        
        apply_background_activity(0.6f);
        
        int spike_count = 0;
        for (int i = 0; i < NEURON_COUNT; ++i) {
            if (neurons[i]->update_and_check_spike()) {
                spike_count++;
                total_spikes++;
                record_spike_event(timestep, i);
            }
        }
        
        timestep++;
    }
    
    sim_data.total_timesteps = timestep;
    sim_data.total_spikes = total_spikes;
}

void NeuronSimulator::apply_metabolic_dysfunction(const MetabolicCondition& condition, int current_timestep) {
    if (current_timestep < condition.onset_timestep) return;
    
    float time_factor = 1.0f;
    if (condition.progressive) {
        time_factor = 1.0f + (current_timestep - condition.onset_timestep) * 0.001f;
        time_factor = std::min(time_factor, 3.0f);
    }
    
    // not used currently
    float atp_factor = condition.atp_efficiency * condition.ion_pump_function / time_factor;
    
    // apply metabolic effects based on condition severity
    if (condition.glucose_level < 50.0f && rand() % 20 == 0) {
        // Hypoglycemia effects
        if (time_factor < 2.0f) {
            std::cout << "Hypoglycemia: Reduced excitability" << std::endl;
        } else if (rand() % 10 == 0) {
            int blocked = rand() % NEURON_COUNT;
            neurons[blocked]->spike();
            std::cout << "Severe hypoglycemia: Depolarization block!" << std::endl;
        }
    }
    
    if (condition.glucose_level > 250.0f && rand() % 15 == 0) {
        // hyperglycemia effects
        for (int burst = 0; burst < 3; ++burst) {
            int affected = rand() % NEURON_COUNT;
            neurons[affected]->spike();
        }
    }
    
    if (condition.atp_efficiency < 0.2f && rand() % 5 == 0) {
        // severe hypoxia
        for (int cascade = 0; cascade < 5; ++cascade) {
            int affected = rand() % NEURON_COUNT;
            neurons[affected]->spike();
        }
    }
}

void NeuronSimulator::run_metabolic_dysfunction_simulation(const MetabolicCondition& condition, int max_timesteps) {
    sim_data.membrane_potentials.clear();
    sim_data.spike_events.clear();
    sim_data.network_activity.clear();
    
    initialize_neurons();
    create_random_connections();
    
    std::cout << "Running " << condition.name << " simulation..." << std::endl;
    
    int timestep = 0;
    bool dysfunction_phase = false;
    
    while (timestep < max_timesteps) {
        if (timestep == condition.onset_timestep && !dysfunction_phase) {
            std::cout << "Metabolic dysfunction onset at timestep " << timestep << std::endl;
            dysfunction_phase = true;
        }
        
        if (dysfunction_phase) {
            apply_metabolic_dysfunction(condition, timestep);
        }
        
        collect_membrane_data();
        
        float stimulation_probability = dysfunction_phase ? 
            std::max(0.1f, 0.5f * condition.atp_efficiency) : 0.5f;
        
        if (static_cast<float>(rand()) / RAND_MAX < stimulation_probability) {
            int stimulated = rand() % NEURON_COUNT;
            neurons[stimulated]->spike();
        }
        
        int timestep_spikes = 0;
        for (int i = 0; i < NEURON_COUNT; ++i) {
            if (neurons[i]->update_and_check_spike()) {
                timestep_spikes++;
                record_spike_event(timestep, i);
            }
        }
        
        timestep++;
    }
    
    sim_data.total_timesteps = timestep;
    sim_data.total_spikes = sim_data.spike_events.size();
}

MetabolicCondition NeuronSimulator::create_hypoglycemia() {
    return {"Severe Hypoglycemia", 35.0f, 0.3f, 0.4f, 0.5f, 0.8f, 2.5f, true, 1000};
}

MetabolicCondition NeuronSimulator::create_diabetes_ketoacidosis() {
    return {"Diabetic Ketoacidosis", 350.0f, 0.6f, 0.3f, 0.4f, 0.6f, 3.5f, true, 800};
}

MetabolicCondition NeuronSimulator::create_hypoxia() {
    return {"Cerebral Hypoxia", 85.0f, 0.1f, 0.2f, 0.3f, 0.5f, 4.0f, true, 500};
}

MetabolicCondition NeuronSimulator::create_mitochondrial_dysfunction() {
    return {"Mitochondrial Dysfunction", 90.0f, 0.4f, 0.6f, 0.7f, 0.7f, 3.0f, false, 200};
}

StabilityMetrics NeuronSimulator::calculate_stability_metrics() const {
    StabilityMetrics metrics = {0};
    
    // simplified implementations for core metrics
    if (!sim_data.spike_events.empty()) {
        // calculate coefficient of variation
        std::vector<float> intervals;
        for (int neuron_id = 0; neuron_id < NEURON_COUNT; ++neuron_id) {
            std::vector<int> neuron_spikes;
            for (const auto& spike : sim_data.spike_events) {
                if (spike.second == neuron_id) {
                    neuron_spikes.push_back(spike.first);
                }
            }
            for (size_t i = 1; i < neuron_spikes.size(); ++i) {
                intervals.push_back(neuron_spikes[i] - neuron_spikes[i-1]);
            }
        }
        
        if (!intervals.empty()) {
            float mean = std::accumulate(intervals.begin(), intervals.end(), 0.0f) / intervals.size();
            float variance = 0.0f;
            for (float interval : intervals) {
                variance += (interval - mean) * (interval - mean);
            }
            variance /= intervals.size();
            metrics.coefficient_of_variation = (mean > 0) ? std::sqrt(variance) / mean : 0.0f;
        }
    }
    
    // calculate homeostatic deviation
    if (!sim_data.network_activity.empty()) {
        float mean_activity = std::accumulate(sim_data.network_activity.begin(), 
                                            sim_data.network_activity.end(), 0.0f) / sim_data.network_activity.size();
        metrics.homeostatic_deviation = std::abs(mean_activity - (-65.0f));
    }
    
    return metrics;
}

void NeuronSimulator::export_csv_data(const std::string& prefix) {
    // export membrane potentials
    std::ofstream mem_file(prefix + "membrane_potentials.csv");
    if (mem_file.is_open()) {
        mem_file << "Timestep";
        for (int i = 0; i < NEURON_COUNT; ++i) {
            mem_file << ",Neuron_" << i;
        }
        mem_file << "\n";
        
        for (size_t t = 0; t < sim_data.membrane_potentials.size(); ++t) {
            mem_file << t;
            for (int i = 0; i < NEURON_COUNT; ++i) {
                mem_file << "," << sim_data.membrane_potentials[t][i];
            }
            mem_file << "\n";
        }
        mem_file.close();
    }
    
    // export spike raster
    std::ofstream spike_file(prefix + "spike_raster.csv");
    if (spike_file.is_open()) {
        spike_file << "Timestep,Neuron_ID\n";
        for (const auto& spike : sim_data.spike_events) {
            spike_file << spike.first << "," << spike.second << "\n";
        }
        spike_file.close();
    }
    
    // export activity summary
    std::ofstream activity_file(prefix + "activity_summary.csv");
    if (activity_file.is_open()) {
        activity_file << "Timestep,Average_Potential,Spike_Count\n";
        
        std::vector<int> spikes_per_timestep(sim_data.total_timesteps, 0);
        for (const auto& spike : sim_data.spike_events) {
            if (spike.first < sim_data.total_timesteps) {
                spikes_per_timestep[spike.first]++;
            }
        }
        
        for (size_t t = 0; t < sim_data.network_activity.size(); ++t) {
            int spike_count = (t < spikes_per_timestep.size()) ? spikes_per_timestep[t] : 0;
            activity_file << t << "," << sim_data.network_activity[t] << "," << spike_count << "\n";
        }
        activity_file.close();
    }
}

void NeuronSimulator::run_metabolic_dysfunction_studies() {
    std::vector<MetabolicCondition> conditions = {
        create_hypoglycemia(),
        create_diabetes_ketoacidosis(),
        create_hypoxia(),
        create_mitochondrial_dysfunction()
    };
    
    std::cout << "Running metabolic dysfunction studies..." << std::endl;
    
    for (size_t i = 0; i < conditions.size(); ++i) {
        std::cout << "\nStudy " << (i+1) << "/" << conditions.size() << ": " << conditions[i].name << std::endl;
        
        run_metabolic_dysfunction_simulation(conditions[i], 2000);
        
        std::string safe_name = conditions[i].name;
        std::replace(safe_name.begin(), safe_name.end(), ' ', '_');
        export_csv_data(safe_name + "_");
        
        StabilityMetrics metrics = calculate_stability_metrics();
        std::cout << "CV: " << metrics.coefficient_of_variation 
                  << ", Homeostatic deviation: " << metrics.homeostatic_deviation << std::endl;
    }
    
    generate_python_visualization("plot_comparison.py");
    std::cout << "\nStudy complete. Run 'python3 plot_comparison.py' to visualize results." << std::endl;
}