# Updated test function with visualization and sequence output
def test_pso_performance_with_visualization(airports):
    configurations = [
        (20, 5000, 500),
        (30, 4000, 400),
        (40, 3000, 300),
        (50, 2000, 200),
        (60, 1000, 100),
        (70, 6000, 600),
    ]

    results = []
    for idx, (particle_count, iterations, iteration_limit) in enumerate(configurations, 1):
        print(f"\nConfiguration {idx}: particle_count={particle_count}, iterations={iterations}, iteration_limit={iteration_limit}")
        best_position, best_fitness, all_fitness_values = particle_swarm_optimization(
            airports, particle_count, iterations, w=1, c1=1, c2=2, velocity_limit=1234.8, iteration_limit=iteration_limit
        )
        results.append((particle_count, iterations, iteration_limit, best_fitness))
        print(f"Best Fitness: {best_fitness:.2f}")

        # Display the sequence of airports
        best_route = [airports.iloc[idx]["name"] for idx in best_position]
        print(f"Best Route Sequence: {' -> '.join(best_route)}")

        # Plot the best route
        route_coordinates = [(airports.iloc[idx]["Lat"], airports.iloc[idx]["Long"]) for idx in best_position]
        x = [coord[0] for coord in route_coordinates]
        y = [coord[1] for coord in route_coordinates]

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x + [x[0]], y + [y[0]], marker='o', linestyle='-')
        plt.title(f'Configuration {idx}: Best Route')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        for airport_name, (xi, yi) in zip(best_route, route_coordinates):
            plt.text(xi, yi, airport_name, fontsize=8, ha='right', va='bottom')

        # Plot the convergence graph
        plt.subplot(1, 2, 2)
        plt.plot(all_fitness_values, marker='o')
        plt.title(f'Configuration {idx}: Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Global Best Fitness')
        plt.tight_layout()
        plt.show()

    return results

# Run tests with visualization
results = test_pso_performance_with_visualization(airports)
for result in results:
    print(f"Configuration: particle_count={result[0]}, iterations={result[1]}, iteration_limit={result[2]} -> Best Fitness: {result[3]:.2f}")
