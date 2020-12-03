def predict_new_state(current_speed, action, time):
    if action == 0:
        new_speed = current_speed + time * ((0.0304 * current_speed**2 + 0.03968 * current_speed - 0.1725) - current_speed)
        distance = time * (current_speed + new_speed)/2

        return distance, new_speed
    if action == 1:
        new_speed = current_speed + time * ((0.01569 * current_speed**2 + 0.4158 * current_speed - 0.2314) - current_speed)
        distance = time * (current_speed + new_speed)/2

        return distance, new_speed
    if action == 2:
        new_speed = current_speed + time * ((0.7269*current_speed + 0.03714) - current_speed)
        distance = time * (current_speed + new_speed)/2

        return distance, new_speed
    if action == 3:
        new_speed = current_speed + time * ((0.883*current_speed + 1.137) - current_speed)
        distance = time * (current_speed + new_speed)/2

        return distance, new_speed
    if action == 4:
        new_speed = current_speed + time * ((1.041*current_speed + 2.532) - current_speed)
        distance = time * (current_speed + new_speed)/2

        return distance, new_speed
