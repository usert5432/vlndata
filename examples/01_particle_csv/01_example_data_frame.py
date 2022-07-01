#!/usr/bin/env python

"""This example parses columnar data from `./data.csv` and prints information
about particles for each event in that file"""

from vlndata.data_frame import CSVFrame

df = CSVFrame(path = './data.csv')

for idx in range(len(df)):
    event_id  = df.get_scalar(column = 'event_id', index = idx)
    particles = df.get_vlarr(column = 'particle_energy', index = idx)

    n_particle = len(particles)
    avg_energy = particles.mean()

    print(
        f'Event {event_id:g}. Number of particles {n_particle}, '
        f'Average energy: {avg_energy:.2f}.'
    )

