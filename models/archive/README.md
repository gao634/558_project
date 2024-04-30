# description

here lies the models that actually work:
    m1: takes in position, heading, env encoding, and goal position for env1 and goes in a straight line
    m2: q learning facing away from goal, 5 points, heading, distance for env2. Eps = 0.9999 and thresh = 0.4, step penalty is 0.2, arrival reward is 1000
    m3: q learning facing towards goal but random, 5 points, heading, distance for env2. Eps = 0.99995 and thresh = 0.4, step penalty is 0.2, arrival reward is 1000
    m4: q learning facing away from goal but random, 5 points, heading, distance for env2. Output is only 3 actions, no going backwards. Eps = 0.99995 and thresh = 0.4, step penalty is 0.2, arrival reward is 1000
    m5: q random, 5 points, heading, distance, velocities (9). for env2. Output is only 3 actions, no going backwards. Eps = 0.99995 and thresh = 0.2, step penalty is 0.2, arrival reward is 1000, angle reward is 10
    m6: random direction, 5 points, heading, distance, velocities (9). Trained from m5 starting at eps=0.5 and 0.99999 decay for 700 epochs. Thresh = 0.2, time penalty = 0.2, arrival reward = 1000, angle reward = 50.
    m7: trained for env 6