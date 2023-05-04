from Simulation import Simulation

env_file = 'SoccerEnvironment.xml'
sim_time = 60 # seconds
players_per_team = 6

sim = Simulation(env_file, sim_time, players_per_team, True)
for i in range(0,1):
	sim.start()
sim.stop()