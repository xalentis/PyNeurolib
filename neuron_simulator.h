#pragma once
#include <vector>
#include <string>
#include <utility>

class Neuron;

struct SimulationData {
    std::vector<std::vector<float>> membrane_potentials;
    std::vector<std::pair<int, int>> spike_events;
    std::vector<int> spikes_per_timestep;
    std::vector<float> network_activity;
    int total_timesteps;
    int total_spikes;
};

struct StabilityMetrics {
    float coefficient_of_variation;
    float burst_coefficient;
    float synchrony_index;
    float entropy;
    float lyapunov_exponent;
    float homeostatic_deviation;
    float network_coherence;
    float critical_branching_ratio;
};

struct MetabolicCondition {
    std::string name;
    float glucose_level;
    float atp_efficiency;
    float ion_pump_function;
    float neurotransmitter_synthesis;
    float membrane_integrity;
    float oxidative_stress;
    bool progressive;
    int onset_timestep;
};

class NeuronSimulator {
public:
    NeuronSimulator();
    ~NeuronSimulator();
    
    // core simulation methods
    void run_standard_simulation(int max_timesteps = 5000);
    void run_metabolic_dysfunction_simulation(const MetabolicCondition& condition, int max_timesteps = 3000);
    void run_metabolic_dysfunction_studies();
    
    // predefined metabolic conditions
    MetabolicCondition create_hypoglycemia();
    MetabolicCondition create_diabetes_ketoacidosis();
    MetabolicCondition create_hypoxia();
    MetabolicCondition create_mitochondrial_dysfunction();
    
    // results
    SimulationData get_simulation_data() const { return sim_data; }
    StabilityMetrics calculate_stability_metrics() const;
    
    // Visualization helpers
    void generate_python_visualization(const std::string& filename);
    void export_csv_data(const std::string& prefix = "");
    
private:
    static const int NEURON_COUNT = 10;
    Neuron* neurons[NEURON_COUNT];
    SimulationData sim_data;
    
    void initialize_neurons();
    void cleanup_neurons();
    void create_random_connections(int connection_density = 6);
    void collect_membrane_data();
    void record_spike_event(int timestep, int neuron_id);
    void apply_background_activity(float noise_probability = 0.3f);
    void apply_metabolic_dysfunction(const MetabolicCondition& condition, int current_timestep);
};