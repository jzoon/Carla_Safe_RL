def predict_new_state(current_speed, current_acc, action, time):
    if action == 0:
        distance = time * (0.7858*current_speed - 0.9539)
        new_speed = current_speed + time * ((0.0304 * current_speed**2 + 0.03968 * current_speed - 0.1725) - current_speed)
        new_acc = current_acc + time * (- 1.476 * current_acc - 4.502)
        return distance, new_speed, new_acc
    if action == 1:
        distance = time * (0.9026*current_speed - 0.8728)
        new_speed = current_speed + time * ((0.01569 * current_speed**2 + 0.4158 * current_speed - 0.2314) - current_speed)
        new_acc = current_acc + time * ( - 1.111*current_acc - 4.537)
        return distance, new_speed, new_acc
    if action == 2:
        distance = time * (0.8561*current_speed + 0.1567)
        new_speed = current_speed + time * ((0.7269*current_speed + 0.03714) - current_speed)
        new_acc = current_acc + time * ( - 1.101*current_acc - 1.199)
        return distance, new_speed, new_acc
    if action == 3:
        distance = time * (0.9543*current_speed + 0.5462)
        new_speed = current_speed + time * ((0.883*current_speed + 1.137) - current_speed)
        new_acc = current_acc + time * ( - 1.022*current_acc + 0.1419)
        return distance, new_speed, new_acc
    if action == 4:
        distance = time * (1.02*current_speed + 1.021)
        new_speed = current_speed + time * ((1.041*current_speed + 2.532) - current_speed)
        new_acc = current_acc + time * ( - 0.8969*current_acc + 3.826)
        return distance, new_speed, new_acc
