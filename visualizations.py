import matplotlib.pyplot as plt
import cv2


def plot_loss(losses, save=False, specs=(0, 0, 0, 0)):
    plt.figure()
    plt.plot(losses)
    plt.title("Momentum: {0} Batch: {1} Rate: {2} Hidden Units: {3}".format(*specs))
    plt.xlabel("Training epochs")
    plt.ylabel("Average Cross Entropy")
    plt.savefig("Img/{0}Mom_{1}Batch_{2}LR_{3}_Hid.png".format(*specs)) if save else plt.show()


def visualize_img(img, scale=5):
    """ Display 32x32x3 RGB image using OpenCV. """
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Display window", img)
    cv2.waitKey(0)

