import neuron_simulator
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def print_menu():
    print("=== Enhanced Human Neuron Network Simulation ===")
    print("Choose simulation mode:")
    print("1. Standard Network Simulation")
    print("2. Metabolic Dysfunction Study")
    print("3. Single Metabolic Condition Test")
    print("4. Exit")


def generate_visualization(csv_prefix=""):

    try:
        membrane_data = pd.read_csv(f'{csv_prefix}membrane_potentials.csv')
        spike_data = pd.read_csv(f'{csv_prefix}spike_raster.csv')
        activity_data = pd.read_csv(f'{csv_prefix}activity_summary.csv')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Neural Network Simulation Results')
        
        # membrane potentials
        for i in range(min(5, len(membrane_data.columns)-1)):
            axes[0,0].plot(membrane_data.iloc[:, i+1], alpha=0.7, label=f'Neuron {i}')
        axes[0,0].set_title('Membrane Potentials')
        axes[0,0].set_ylabel('Potential (mV)')
        axes[0,0].legend()
        
        # spike raster
        if len(spike_data) > 0:
            axes[0,1].scatter(spike_data['Timestep'], spike_data['Neuron_ID'], s=1, alpha=0.6)
            axes[0,1].set_title('Spike Raster')
            axes[0,1].set_ylabel('Neuron ID')
        
        # network activity
        axes[1,0].plot(activity_data['Timestep'], activity_data['Average_Potential'])
        axes[1,0].set_title('Network Activity')
        axes[1,0].set_ylabel('Average Potential (mV)')
        axes[1,0].set_xlabel('Timestep')
        
        # spike rate
        axes[1,1].plot(activity_data['Timestep'], activity_data['Spike_Count'])
        axes[1,1].set_title('Spike Rate')
        axes[1,1].set_ylabel('Spikes per Timestep')
        axes[1,1].set_xlabel('Timestep')
        
        plt.tight_layout()
        
        output_filename = f'{csv_prefix}simulation_results.png' if csv_prefix else 'simulation_results.png'
        plt.savefig(output_filename, dpi=300)
        print(f"Visualization saved as: {output_filename}")
        
        plt.show()
        
    except FileNotFoundError as e:
        print(f'Error: {e}')
        print('Make sure to export CSV data first.')
    except Exception as e:
        print(f'Error creating visualization: {e}')

def run_standard_simulation():
    print("Running standard neural network simulation...")
    
    simulator = neuron_simulator.NeuronSimulator()
    simulator.run_standard_simulation(5000)
    simulator.export_csv_data()
    metrics = simulator.calculate_stability_metrics()
    data = simulator.get_simulation_data()
    
    print(f"\nSimulation Complete:")
    print(f"Total timesteps: {data.total_timesteps}")
    print(f"Total spikes: {data.total_spikes}")
    print(f"Spike rate: {data.total_spikes / data.total_timesteps:.3f} spikes/timestep")
    print(f"Coefficient of variation: {metrics.coefficient_of_variation:.3f}")
    print(f"Synchrony index: {metrics.synchrony_index:.3f}")
    print(f"Network entropy: {metrics.entropy:.3f}")
    print(f"Homeostatic deviation: {metrics.homeostatic_deviation:.1f} mV")
    
    print("\nGenerating visualization...")
    generate_visualization()

def run_metabolic_studies():
    print("Running comprehensive metabolic dysfunction studies...")
    
    simulator = neuron_simulator.NeuronSimulator()
    simulator.run_metabolic_dysfunction_studies()
    
    print("Metabolic studies complete!")
    print("Generating comparative visualizations...")
    
    conditions = ['Severe_Hypoglycemia_', 'Diabetic_Ketoacidosis_', 'Cerebral_Hypoxia_', 'Mitochondrial_Dysfunction_']
    for condition_prefix in conditions:
        try:
            generate_visualization(condition_prefix)
        except:
            print(f"Could not generate visualization for {condition_prefix}")

def run_single_condition():
    print("Available metabolic conditions:")
    print("1. Severe Hypoglycemia")
    print("2. Diabetic Ketoacidosis") 
    print("3. Cerebral Hypoxia")
    print("4. Mitochondrial Dysfunction")
    
    try:
        choice = int(input("Enter condition (1-4): "))
    except ValueError:
        print("Invalid input. Using hypoglycemia.")
        choice = 1
    
    simulator = neuron_simulator.NeuronSimulator()
    
    if choice == 1:
        condition = simulator.create_hypoglycemia()
    elif choice == 2:
        condition = simulator.create_diabetes_ketoacidosis()
    elif choice == 3:
        condition = simulator.create_hypoxia()
    elif choice == 4:
        condition = simulator.create_mitochondrial_dysfunction()
    else:
        print("Invalid choice, using hypoglycemia.")
        condition = simulator.create_hypoglycemia()
    
    print(f"\nRunning {condition.name} simulation...")
    print(f"Glucose level: {condition.glucose_level} mg/dL")
    print(f"ATP efficiency: {condition.atp_efficiency * 100:.1f}%")
    print(f"Ion pump function: {condition.ion_pump_function * 100:.1f}%")
    print(f"Dysfunction onset: Timestep {condition.onset_timestep}")
    
    simulator.run_metabolic_dysfunction_simulation(condition, 3000)
    
    safe_name = condition.name.replace(' ', '_').replace('(', '_').replace(')', '_')
    simulator.export_csv_data(f"{safe_name}_")
    metrics = simulator.calculate_stability_metrics()
    data = simulator.get_simulation_data()
    
    print(f"\n=== {condition.name} Results ===")
    print(f"Total spikes: {data.total_spikes}")
    print(f"Coefficient of variation: {metrics.coefficient_of_variation:.3f}")
    print(f"Synchrony index: {metrics.synchrony_index:.3f}")
    print(f"Homeostatic deviation: {metrics.homeostatic_deviation:.1f} mV")
    
    # clinical interpretation
    spike_rate = data.total_spikes / data.total_timesteps if data.total_timesteps > 0 else 0
    if spike_rate < 0.1:
        print("SEVERE DYSFUNCTION: Network activity critically reduced")
    elif spike_rate > 0.5:
        print("HYPEREXCITABILITY: Network showing seizure-like activity")
    else:
        print("MILD DYSFUNCTION: Network partially compensating")
    
    print(f"Data exported with prefix: {safe_name}_")
    print("Generating visualization...")
    generate_visualization(f"{safe_name}_")

def main():
    while True:
        print_menu()
        
        try:
            choice = int(input("\nEnter choice (1-4): "))
        except ValueError:
            print("Invalid input. Please enter a number 1-4.")
            continue
        
        if choice == 1:
            run_standard_simulation()
        elif choice == 2:
            run_metabolic_studies()
        elif choice == 3:
            run_single_condition()
        elif choice == 4:
            print("Exiting simulation...")
            sys.exit(0)
        else:
            print("Invalid choice. Please select 1-4.")
        
        input("\nPress Enter to continue...")
        print()

if __name__ == "__main__":
    main()