from src import *
import GPy
import numpy as np
import copy
import torch
import matplotlib
import matplotlib.pyplot as plt

log = simple_logging.getLogger()


def main():
    utils.init()
    x, y = data_source.fetch_data('600519.SH', 'open')
    np.random.seed(2)
    no_points = len(x)
    lengthscale = 1
    variance = 0
    sig_noise = 0.3
    k = GPy.kern.RBF(input_dim=1, variance=variance, lengthscale=lengthscale)
    C = k.K(x, x) + np.eye(no_points) * sig_noise ** 2
    x, y = (x - x.mean()), (y - y.mean())
    x_train = x[1000:3500]
    y_train = y[1000:3500]
    x_mean = x_train.mean()
    y_mean = y_train.mean()
    x_std = x_train.var() ** 0.5
    y_std = y_train.var() ** 0.5
    x_train = (x_train - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    num_epochs = 3500
    batch_size = nb_train = len(x_train)
    learning_rate = 1e-1
    mini_batch_size = 10
    display_interval = 100
    net = models.BBP_Homoscedastic_Model_Wrapper(input_dim=1, output_dim=1, no_units=100, learn_rate=learning_rate,
                                                 batch_size=batch_size, no_batches=1, init_log_noise=0)
    fit_loss_train = np.zeros(num_epochs)
    KL_loss_train = np.zeros(num_epochs)
    total_loss = np.zeros(num_epochs)
    best_net, best_loss = None, float('inf')
    for i in range(num_epochs):
        fit_loss, KL_loss = net.fit(x_train, y_train, no_samples=mini_batch_size)
        fit_loss_train[i] = fit_loss.cpu().data.numpy()
        KL_loss_train[i] = KL_loss.cpu().data.numpy()
        total_loss[i] = fit_loss_train[i] + KL_loss_train[i]
        if fit_loss < best_loss:
            best_loss = fit_loss
            best_net = copy.deepcopy(net.network)
        if i % display_interval == 0 or i == num_epochs - 1:
            log.info("Epoch: %5d/%5d, Fit loss = %8.3f, KL loss = %8.3f, noise = %6.3f" %
                     (i + 1, num_epochs, fit_loss_train[i], KL_loss_train[i],
                      net.network.log_noise.exp().cpu().data.numpy()))
    plt.plot(range(num_epochs), fit_loss_train, c='r')
    plt.plot(range(num_epochs), KL_loss_train, c='b')
    plt.plot(range(num_epochs), total_loss, c='g')
    plt.show()
    sampling_count = 100
    sampling_point_num = 2000
    sampling_bottom, sampling_top = -5, 5
    samples = []
    for i in range(sampling_count):
        preds = (best_net.forward(torch.linspace(sampling_bottom, sampling_top, sampling_point_num).cuda())[0] * y_std) + y_mean
        samples.append(preds.cpu().data.numpy()[:, 0])
    samples = np.array(samples)
    means = samples.mean(axis=0)

    aleatoric = best_net.log_noise.exp().cpu().data.numpy()
    epistemic = samples.var(axis=0) ** 0.5
    total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5

    c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    plt.figure(figsize=(6, 5))
    plt.style.use('default')
    plt.scatter((x_train * x_std) + x_mean, (y_train * y_std) + y_mean, s=10, marker='x', color='black', alpha=0.5)
    plt.fill_between(np.linspace(sampling_bottom, sampling_top, sampling_point_num) * x_std + x_mean, means + aleatoric, means + total_unc, color=c[0],
                     alpha=0.3, label=r'$\sigma(y^*|x^*)$')
    plt.fill_between(np.linspace(sampling_bottom, sampling_top, sampling_point_num) * x_std + x_mean, means - total_unc, means - aleatoric, color=c[0],
                     alpha=0.3)
    plt.fill_between(np.linspace(sampling_bottom, sampling_top, sampling_point_num) * x_std + x_mean, means - aleatoric, means + aleatoric, color=c[1],
                     alpha=0.4, label=r'$\EX[\sigma^2]^{1/2}$')
    plt.plot(np.linspace(-5, 5, 2000) * x_std + x_mean, means, color='black', linewidth=1)
    plt.xlim([-2300, 2300])
    # plt.ylim([-5, 7])
    # plt.ylim(bottom=0)
    plt.ylim([-210, 50])
    plt.xlabel('$x$', fontsize=3)
    plt.title('BBP', fontsize=40)
    plt.tick_params(labelsize=30)
    # plt.xticks(np.arange(-4, 5, 2))
    # plt.gca().set_yticklabels([])
    plt.gca().yaxis.grid(alpha=0.3)
    plt.gca().xaxis.grid(alpha=0.3)
    plt.savefig('bbp_homo.pdf', bbox_inches='tight')

    # files.download("bbp_homo.pdf")

    plt.show()


if __name__ == '__main__':
    main()
