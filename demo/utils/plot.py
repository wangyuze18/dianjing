from matplotlib import pyplot as plt

def set_figsize(figsize=(3.5, 2.5)):  # @save
    """设置matplotlib的图表大小"""
    plt.rcParams['figure.figsize'] = figsize


# @save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


# @save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    # 通过数据在空白画布轴上画画，（x数据，y数据，线的样式）
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    # 将空白画布裱进画框里，设置画框的参数（画布，x轴名称，y轴名称，x轴的范围，y轴的范围，线的名称）
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5,cmap=None):  # @save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    """使用了plt.subplots()函数创建了一个包含num_rows行、num_cols列的子图网格，
    并将返回的figure对象赋值给_，将所有子图对象赋值给axes.
    _是一个占位符变量，一般用于表示不需要使用的变量。"""
    axes = axes.flatten()
    # flatten()方法将axes数组展平为一维数组，便于之后对子图进行索引和操作
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # PIL图片
        ax.imshow(img,cmap=cmap)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        # 隐藏当前子图的X轴和Y轴刻度
        if titles:
            ax.set_title(titles[i])
    return axes
