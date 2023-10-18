def save_loss_history(loss_history, filename):
    with open(filename, 'w') as file:
        for item in loss_history:
            file.write(str(item) + '\n')