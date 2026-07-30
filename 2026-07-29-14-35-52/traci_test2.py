import traci

# Point to your config file
SUMO_CONFIG = "osm.sumocfg"

# Use sumo-gui so you can watch it happen visually
sumo_cmd = ["sumo-gui", "-c", SUMO_CONFIG]

# Start the simulation
traci.start(sumo_cmd)
print("Connected to SUMO successfully")

# Run for 200 steps and print queue data
for step in range(200):
    traci.simulationStep()

    # Get all lane IDs in the network
    lanes = traci.lane.getIDList()

    # Every 50 steps print the halting vehicle count on each lane
    if step % 50 == 0:
        print(f"\n--- Step {step} ---")
        for lane in lanes:
            halting = traci.lane.getLastStepHaltingNumber(lane)
            if halting > 0:  # only print lanes with queuing vehicles
                print(f"  Lane {lane}: {halting} vehicles waiting")

# Close the connection
traci.close()
print("\nSimulation complete")