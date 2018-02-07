import os, imageio
import matplotlib.pyplot as plt


def create_dir(path=''):
    if not os.path.isdir(path + '/Random_results'):
        os.makedirs(path + '/Random_results')
    if not os.path.isdir(path + '/Fixed_results'):
        os.makedirs(path + '/Fixed_results')


def save_result(image, num_epoch, show=False, save=False, path='result.png'):
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))

    for i in range(size_figure_grid):
        for j in range(size_figure_grid):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        if image[k, :].cpu().data.numpy().size == 784:
            ax[i, j].imshow(image[k, :].cpu().data.view(28, 28).numpy(), cmap='gray')
        else:
            ax[i, j].imshow(image[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def save_history(hist, show=False, save=False, path='history.png'):
    x = range(len(hist['D_losses']))
    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_losses')
    plt.plot(x, y2, label='G_losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def save_animation(num_epoch, prefix, path):
    images = []
    for e in range(num_epoch):
        img_name = prefix + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path, images, fps=10)
