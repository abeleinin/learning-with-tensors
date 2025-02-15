import mlx.core as mx
import matplotlib.pyplot as plt

user_points = []

gp_line = None
fill_2sigma = None
posterior_samples = []

# rbf kernel function
def rbf_kernel(Xa, Xb, length_scale=1.0):
    Xa = mx.array(Xa).reshape(-1, 1)
    Xb = mx.array(Xb).reshape(-1, 1)
    sqdist = (Xa - Xb.T)**2
    return mx.exp(-0.5 * sqdist / length_scale**2)

# gaussian process posterior
def GP(X1, y1, X2, kernel_func):
    Sigma11 = kernel_func(X1, X1)
    Sigma12 = kernel_func(X1, X2)

    # solve for (Sigma11^-1 * Sigma12)
    solved = mx.linalg.solve(Sigma11, Sigma12, stream=mx.cpu).T

    # posterior mean
    mu_post = solved @ y1

    # posterior covariance
    Sigma_post = kernel_func(X2, X2) - (solved @ Sigma12)

    return mu_post, Sigma_post

# mouse click event
def onclick(event):
    global user_points, gp_line, fill_2sigma, posterior_samples

    if event.button == 1 and event.inaxes is not None:
        x, y = event.xdata, event.ydata
        user_points.append((round(x, 3), round(y, 3)))

        point_label = '$(x_1, y_1)$' if len(user_points) == 1 else None
        plt.plot(x, y, 'ro', label=point_label)

        if len(user_points) > 0:
            X1 = mx.array([p[0] for p in user_points])
            y1 = mx.array([p[1] for p in user_points])

            X2 = mx.linspace(-10, 10, 200)

            # compute gp posterior mean & covariance
            mu, Sigma2 = GP(X1, y1, X2, rbf_kernel)

            # clear graph
            if gp_line is not None:
                gp_line.remove()
                gp_line = None
            if fill_2sigma is not None:
                fill_2sigma.remove()
                fill_2sigma = None
            for smp_line in posterior_samples:
                smp_line.remove()
            posterior_samples = []

            # plot gp mean
            gp_line, = ax.plot(X2, mu, 'b-', label='GP mean')

            # calculate 2 sigma confidence interval
            std = mx.sqrt(mx.diag(Sigma2))
            upper = mu + 2 * std
            lower = mu - 2 * std

            # fill in confidence interval
            fill_2sigma = ax.fill_between(
                X2, lower, upper,
                color='red', alpha=0.15, label='$2 \sigma$'
            )

            # sample multiple posterior draws from N(mu, Sigma2)
            num_samples = 5
            for i in range(num_samples):
                sample_i = mx.random.multivariate_normal(mu, Sigma2, stream=mx.cpu)
                posterior_label = 'Posterior samples' if i == 0 else None
                smp_line, = ax.plot(X2, sample_i, color='orange', alpha=0.3, label=posterior_label)
                posterior_samples.append(smp_line)

        ax.legend(loc='best')
        plt.draw()

# plot figure
fig, ax = plt.subplots()
ax.set_xlim([10, -10])
ax.set_ylim([10, -10])
ax.legend()

ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

ax.set_title('Interactive MLX Gaussian Process')
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.tight_layout()
plt.show()
