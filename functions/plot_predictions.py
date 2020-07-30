import matplotlib.pyplot as plt
import numpy as np

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array) 
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    class_names = ["Fluorescent", "Purple", "Grayscale", "Pink-Purple"]

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[int(predicted_label)],
                                    100*np.max(predictions_array),
                                    class_names[int(true_label)]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(4))
    plt.yticks([])
    thisplot = plt.bar(range(4), predictions_array, color="#777777", width=0.5)
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[int(true_label)].set_color('blue')
    
