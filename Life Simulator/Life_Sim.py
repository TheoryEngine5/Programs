import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import json
import time
from datetime import datetime
from copy import deepcopy
import threading
import queue

# -----------------------------
# ENHANCED CONFIG
# -----------------------------
class EvolutionConfig:
    def __init__(self):
        self.N = 60                    # grid size
        self.steps = 500               # simulation length
        self.num_runs = 10             # number of planets
        self.mutation_rate = 0.02      # base mutation rate
        self.output_dir = "multiverse_evolution"
        
        # Dynamic environment parameters
        self.nutrient_regen_rate = 0.01
        self.abiogenesis_rate = 0.0005
        self.carrying_capacity_factor = 0.8
        self.environmental_pressure = 0.1
        
        # Evolution parameters
        self.genome_complexity = 5     # number of genome traits
        self.trait_bounds = [(0.01, 1.0), (0.01, 0.5), (10, 100), (0.01, 1.0), (1, 50)]
        
        # Self-evolution parameters
        self.self_evolution_rate = 0.05
        self.parameter_drift_rate = 0.001

# -----------------------------
# ENHANCED GENOME SYSTEM
# -----------------------------
class EnhancedGenome:
    def __init__(self, config):
        self.config = config
        self.traits = self.generate_random_genome()
        self.fitness_history = deque(maxlen=10)
        self.age = 0
        self.generation = 0
        self.lineage_id = 0
        self.mutations_accumulated = 0
        
    def generate_random_genome(self):
        """Generate random genome with enhanced traits"""
        return [
            random.uniform(*self.config.trait_bounds[0]),  # survival_rate
            random.uniform(*self.config.trait_bounds[1]),  # reproduction_rate  
            random.randint(*self.config.trait_bounds[2]),  # max_lifespan
            random.uniform(*self.config.trait_bounds[3]),  # metabolic_efficiency
            random.randint(*self.config.trait_bounds[4])   # cooperation_factor
        ]
    
    def mutate(self, environmental_pressure=0.1):
        """Advanced mutation with environmental adaptation"""
        new_traits = self.traits[:]
        mutation_strength = self.config.mutation_rate * (1 + environmental_pressure)
        
        for i, (lower, upper) in enumerate(self.config.trait_bounds):
            if random.random() < mutation_strength:
                if isinstance(new_traits[i], float):
                    # Gaussian mutation for continuous traits
                    mutation = random.gauss(0, (upper - lower) * 0.1)
                    new_traits[i] += mutation
                    new_traits[i] = max(lower, min(upper, new_traits[i]))
                else:
                    # Integer mutation for discrete traits
                    mutation = random.choice([-5, -2, -1, 1, 2, 5])
                    new_traits[i] += mutation
                    new_traits[i] = max(int(lower), min(int(upper), new_traits[i]))
                
                self.mutations_accumulated += 1
        
        # Create offspring genome
        offspring = EnhancedGenome(self.config)
        offspring.traits = new_traits
        offspring.generation = self.generation + 1
        offspring.lineage_id = self.lineage_id
        return offspring
    
    def fitness_score(self, environment_data):
        """Calculate fitness based on genome and environment"""
        base_fitness = sum(self.traits) / len(self.traits)
        
        # Environmental adaptation bonus
        adaptation_bonus = 0
        if environment_data:
            if environment_data.get('nutrient_abundance', 0.5) < 0.3:
                # Scarce resources favor efficiency
                adaptation_bonus += self.traits[3] * 0.5  # metabolic_efficiency
            if environment_data.get('population_density', 0.5) > 0.7:
                # High density favors cooperation
                adaptation_bonus += self.traits[4] / 50.0  # cooperation_factor
        
        return base_fitness + adaptation_bonus

class EnhancedCell:
    def __init__(self, state=0, genome=None, age=0, energy=100):
        self.state = state              # 0=empty, 1=nutrient, 2=life
        self.genome = genome
        self.age = age
        self.energy = energy
        self.stress_level = 0.0
        self.social_connections = []
        
    def can_reproduce(self, local_environment):
        """Enhanced reproduction conditions"""
        if not self.genome or self.state != 2:
            return False
        
        # Energy threshold
        if self.energy < 50:
            return False
        
        # Social cooperation bonus
        cooperation_bonus = min(len(self.social_connections) * 0.1, 0.5)
        reproduction_prob = self.genome.traits[1] + cooperation_bonus
        
        return random.random() < reproduction_prob

class SelfEvolvingPlanet:
    def __init__(self, seed, planet_id, config):
        self.seed = seed
        self.planet_id = planet_id
        self.config = config
        self.grid = None
        self.next_lineage = 0
        
        # Enhanced tracking
        self.history = {
            'population': [],
            'diversity': [],
            'avg_fitness': [],
            'environmental_pressure': [],
            'cooperation_index': [],
            'mutation_rate': [],
            'extinction_events': [],
            'speciation_events': []
        }
        
        self.lineage_tracker = defaultdict(lambda: {
            'count': 0,
            'fitness_sum': 0,
            'generations_survived': 0,
            'last_seen': 0
        })
        
        self.environment_state = {
            'nutrient_abundance': 0.5,
            'population_density': 0.0,
            'genetic_diversity': 0.0,
            'competition_level': 0.0
        }
        
        # Self-evolution tracking
        self.evolution_log = []
        
    def initialize_grid(self):
        """Initialize planet with enhanced environmental distribution"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.grid = []
        for i in range(self.config.N):
            row = []
            for j in range(self.config.N):
                # Create environmental gradients
                distance_from_center = np.sqrt((i - self.config.N//2)**2 + (j - self.config.N//2)**2)
                nutrient_prob = 0.3 * (1 - distance_from_center / (self.config.N//2))
                nutrient_prob = max(0.1, min(0.5, nutrient_prob))
                
                cell_state = np.random.choice([0, 1], p=[1-nutrient_prob, nutrient_prob])
                row.append(EnhancedCell(state=cell_state))
            self.grid.append(row)
    
    def get_local_environment(self, i, j, radius=2):
        """Analyze local environment around a cell"""
        local_data = {
            'population_count': 0,
            'nutrient_count': 0,
            'avg_fitness': 0.0,
            'cooperation_level': 0.0,
            'genetic_diversity': set()
        }
        
        fitness_sum = 0
        cooperation_sum = 0
        
        for di in range(-radius, radius+1):
            for dj in range(-radius, radius+1):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.config.N and 0 <= nj < self.config.N:
                    cell = self.grid[ni][nj]
                    
                    if cell.state == 1:
                        local_data['nutrient_count'] += 1
                    elif cell.state == 2 and cell.genome:
                        local_data['population_count'] += 1
                        fitness_sum += cell.genome.fitness_score(self.environment_state)
                        cooperation_sum += cell.genome.traits[4]
                        local_data['genetic_diversity'].add(tuple(cell.genome.traits))
        
        if local_data['population_count'] > 0:
            local_data['avg_fitness'] = fitness_sum / local_data['population_count']
            local_data['cooperation_level'] = cooperation_sum / local_data['population_count']
        
        local_data['genetic_diversity'] = len(local_data['genetic_diversity'])
        return local_data
    
    def self_evolve_parameters(self, step):
        """Self-modify simulation parameters based on outcomes"""
        if step > 50 and step % 50 == 0:  # Every 50 steps after warmup
            population = self.history['population'][-10:]  # Last 10 steps
            diversity = self.history['diversity'][-10:]
            
            # Analyze trends
            pop_trend = np.polyfit(range(len(population)), population, 1)[0]
            div_trend = np.polyfit(range(len(diversity)), diversity, 1)[0]
            
            modifications = []
            
            # Adjust mutation rate based on diversity trends
            if div_trend < -0.1:  # Diversity declining
                old_rate = self.config.mutation_rate
                self.config.mutation_rate *= 1.1
                modifications.append(f"Increased mutation rate: {old_rate:.4f} → {self.config.mutation_rate:.4f}")
            elif div_trend > 0.5:  # Diversity increasing too fast
                old_rate = self.config.mutation_rate
                self.config.mutation_rate *= 0.95
                modifications.append(f"Decreased mutation rate: {old_rate:.4f} → {self.config.mutation_rate:.4f}")
            
            # Adjust environmental pressure
            if pop_trend < -1:  # Population declining
                old_pressure = self.config.environmental_pressure
                self.config.environmental_pressure *= 0.9
                modifications.append(f"Reduced environmental pressure: {old_pressure:.3f} → {self.config.environmental_pressure:.3f}")
            elif pop_trend > 2:  # Population growing too fast
                old_pressure = self.config.environmental_pressure
                self.config.environmental_pressure *= 1.1
                modifications.append(f"Increased environmental pressure: {old_pressure:.3f} → {self.config.environmental_pressure:.3f}")
            
            if modifications:
                evolution_event = {
                    'step': step,
                    'modifications': modifications,
                    'trigger': f'pop_trend={pop_trend:.2f}, div_trend={div_trend:.2f}'
                }
                self.evolution_log.append(evolution_event)
                print(f"🧬 Planet {self.planet_id} self-evolved at step {step}:")
                for mod in modifications[:2]:  # Show first 2 modifications
                    print(f"   {mod}")
    
    def step(self, step_num):
        """Enhanced simulation step with environmental dynamics"""
        new_grid = [[EnhancedCell() for _ in range(self.config.N)] for _ in range(self.config.N)]
        
        # Global statistics
        population = 0
        total_fitness = 0
        cooperation_sum = 0
        genetic_diversity = set()
        current_lineages = defaultdict(int)
        
        # Environmental update
        self.update_environment_state()
        
        for i in range(self.config.N):
            for j in range(self.config.N):
                cell = self.grid[i][j]
                local_env = self.get_local_environment(i, j)
                
                # Empty cell
                if cell.state == 0:
                    # Nutrient regeneration with environmental influence
                    regen_rate = self.config.nutrient_regen_rate * (1 + self.environment_state['nutrient_abundance'])
                    if random.random() < regen_rate:
                        new_grid[i][j] = EnhancedCell(state=1)
                    else:
                        new_grid[i][j] = EnhancedCell(state=0)
                
                # Nutrient cell
                elif cell.state == 1:
                    # Abiogenesis with environmental factors
                    abiogenesis_rate = self.config.abiogenesis_rate
                    if local_env['population_count'] == 0:  # Higher chance in empty areas
                        abiogenesis_rate *= 2
                    
                    if random.random() < abiogenesis_rate:
                        new_genome = EnhancedGenome(self.config)
                        new_genome.lineage_id = self.next_lineage
                        new_cell = EnhancedCell(state=2, genome=new_genome)
                        new_grid[i][j] = new_cell
                        self.next_lineage += 1
                    else:
                        new_grid[i][j] = EnhancedCell(state=1)
                
                # Living cell
                elif cell.state == 2 and cell.genome:
                    cell.age += 1
                    genome = cell.genome
                    
                    # Update statistics
                    population += 1
                    fitness = genome.fitness_score(self.environment_state)
                    total_fitness += fitness
                    cooperation_sum += genome.traits[4]
                    genetic_diversity.add(tuple(genome.traits))
                    current_lineages[genome.lineage_id] += 1
                    
                    # Energy and stress dynamics
                    cell.energy -= 5  # Base metabolic cost
                    cell.energy += local_env['nutrient_count'] * genome.traits[3] * 10  # Metabolism
                    cell.stress_level = local_env['population_count'] / 20.0  # Crowding stress
                    
                    # Death conditions
                    death_prob = 0
                    if cell.age > genome.traits[2]:  # Natural lifespan
                        death_prob = 0.8
                    if cell.energy <= 0:  # Starvation
                        death_prob = 0.9
                    if cell.stress_level > 0.8:  # Extreme stress
                        death_prob += 0.3
                    
                    if random.random() < death_prob:
                        new_grid[i][j] = EnhancedCell(state=1)  # Becomes nutrient
                        continue
                    
                    # Reproduction
                    if cell.can_reproduce(local_env) and cell.energy > 60:
                        neighbors = [(i+di, j+dj) for di in [-1,0,1] for dj in [-1,0,1] 
                                   if (di,dj) != (0,0)]
                        random.shuffle(neighbors)
                        
                        for ni, nj in neighbors:
                            if (0 <= ni < self.config.N and 0 <= nj < self.config.N and 
                                new_grid[ni][nj].state == 0):
                                
                                # Create offspring with enhanced mutation
                                offspring_genome = genome.mutate(self.config.environmental_pressure)
                                offspring_cell = EnhancedCell(
                                    state=2, 
                                    genome=offspring_genome,
                                    energy=cell.energy // 2
                                )
                                new_grid[ni][nj] = offspring_cell
                                cell.energy //= 2  # Parent loses energy
                                break
                    
                    # Parent survives
                    new_grid[i][j] = EnhancedCell(
                        state=2, 
                        genome=genome,
                        age=cell.age,
                        energy=cell.energy
                    )
        
        self.grid = new_grid
        
        # Update lineage tracking
        for lineage_id, count in current_lineages.items():
            self.lineage_tracker[lineage_id]['count'] = count
            self.lineage_tracker[lineage_id]['last_seen'] = step_num
            if count > 0:
                self.lineage_tracker[lineage_id]['generations_survived'] += 1
        
        # Record history
        self.history['population'].append(population)
        self.history['diversity'].append(len(genetic_diversity))
        self.history['avg_fitness'].append(total_fitness / max(1, population))
        self.history['cooperation_index'].append(cooperation_sum / max(1, population))
        self.history['mutation_rate'].append(self.config.mutation_rate)
        self.history['environmental_pressure'].append(self.config.environmental_pressure)
        
        # Self-evolution check
        self.self_evolve_parameters(step_num)
        
        return population
    
    def update_environment_state(self):
        """Update global environment state"""
        if self.history['population']:
            recent_pop = self.history['population'][-10:]
            recent_div = self.history['diversity'][-10:]
            
            self.environment_state.update({
                'population_density': np.mean(recent_pop) / (self.config.N * self.config.N),
                'genetic_diversity': np.mean(recent_div) / max(1, max(recent_div) if recent_div else 1),
                'competition_level': min(1.0, np.mean(recent_pop) / 1000),
                'nutrient_abundance': max(0.1, 1.0 - self.environment_state['population_density'])
            })
    
    def run_simulation(self):
        """Run enhanced simulation with real-time adaptation"""
        print(f"🌍 Initializing Planet {self.planet_id} with self-evolution...")
        self.initialize_grid()
        
        log_data = []
        for step in range(self.config.steps):
            population = self.step(step)
            
            # Detailed logging
            log_data.append({
                "step": step,
                "population": population,
                "diversity": self.history['diversity'][-1],
                "avg_fitness": self.history['avg_fitness'][-1],
                "cooperation_index": self.history['cooperation_index'][-1],
                "mutation_rate": self.history['mutation_rate'][-1],
                "environmental_pressure": self.history['environmental_pressure'][-1]
            })
            
            # Progress updates
            if step % 100 == 0:
                print(f"   Step {step}: Pop={population}, Div={self.history['diversity'][-1]}, "
                      f"Evolutions={len(self.evolution_log)}")
        
        # Save detailed results
        os.makedirs(self.config.output_dir, exist_ok=True)
        csv_file = f"{self.config.output_dir}/planet_{self.planet_id}_enhanced.csv"
        pd.DataFrame(log_data).to_csv(csv_file, index=False)
        
        # Save evolution log
        evolution_file = f"{self.config.output_dir}/planet_{self.planet_id}_evolution_log.json"
        with open(evolution_file, 'w') as f:
            json.dump(self.evolution_log, f, indent=2)
        
        return self.get_enhanced_summary()
    
    def get_enhanced_summary(self):
        """Generate comprehensive summary with self-evolution metrics"""
        pop_hist = self.history['population']
        div_hist = self.history['diversity']
        fitness_hist = self.history['avg_fitness']
        coop_hist = self.history['cooperation_index']
        
        # Advanced analysis
        evolutionary_stability = np.std(div_hist[-50:]) if len(div_hist) >= 50 else 0
        cooperation_evolution = coop_hist[-1] - coop_hist[0] if len(coop_hist) > 1 else 0
        adaptation_events = len(self.evolution_log)
        
        return {
            "planet_id": self.planet_id,
            "max_population": max(pop_hist) if pop_hist else 0,
            "max_diversity": max(div_hist) if div_hist else 0,
            "final_population": pop_hist[-1] if pop_hist else 0,
            "avg_fitness": np.mean(fitness_hist) if fitness_hist else 0,
            "cooperation_evolution": cooperation_evolution,
            "evolutionary_stability": evolutionary_stability,
            "adaptation_events": adaptation_events,
            "successful_lineages": len([l for l in self.lineage_tracker.values() if l['count'] > 5]),
            "went_extinct": (pop_hist[-1] if pop_hist else 0) == 0,
            "final_mutation_rate": self.config.mutation_rate,
            "final_environmental_pressure": self.config.environmental_pressure
        }

def run_enhanced_multiverse(num_planets=10, steps=500):
    """Run the enhanced self-evolving multiverse simulation"""
    print("🌌 SELF-EVOLVING MULTIVERSE DIGITAL LIFE SIMULATOR")
    print("=" * 70)
    
    config = EvolutionConfig()
    config.num_runs = num_planets
    config.steps = steps
    
    planets = []
    summaries = []
    
    # Run simulations
    for planet_id in range(1, num_planets + 1):
        planet_config = deepcopy(config)  # Each planet gets its own config
        planet = SelfEvolvingPlanet(seed=planet_id, planet_id=planet_id, config=planet_config)
        
        summary = planet.run_simulation()
        planets.append(planet)
        summaries.append(summary)
        
        status = "EXTINCT" if summary['went_extinct'] else "EVOLVED"
        print(f"✅ Planet {planet_id}: {status} | Adaptations: {summary['adaptation_events']} | "
              f"Final Pop: {summary['final_population']}")
    
    # Analysis and reporting
    summary_df = pd.DataFrame(summaries)
    
    print("\n" + "="*70)
    print("🌌 ENHANCED MULTIVERSE EVOLUTION REPORT")
    print("="*70)
    print(f"📊 Total Planets: {num_planets}")
    print(f"🧬 Self-Evolution Events: {summary_df['adaptation_events'].sum()}")
    print(f"💀 Extinction Rate: {summary_df['went_extinct'].sum()}/{num_planets} "
          f"({100*summary_df['went_extinct'].mean():.1f}%)")
    print(f"🤝 Average Cooperation Evolution: {summary_df['cooperation_evolution'].mean():.3f}")
    print(f"⚖️  Average Evolutionary Stability: {summary_df['evolutionary_stability'].mean():.3f}")
    print(f"🏆 Most Adaptive Planet: #{summary_df.loc[summary_df['adaptation_events'].idxmax(), 'planet_id']}")
    
    # Save comprehensive results
    summary_file = f"{config.output_dir}/enhanced_multiverse_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n📁 Results saved to: {config.output_dir}/")
    
    return summary_df, planets

if __name__ == "__main__":
    # Run the enhanced simulation
    summary_df, planets = run_enhanced_multiverse(num_planets=8, steps=400)
    print("\n✨ Enhanced multiverse simulation complete!")