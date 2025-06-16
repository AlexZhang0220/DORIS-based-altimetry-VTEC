import matplotlib.pyplot as plt

# 假设你已经有以下两个 numpy array 或 pandas Series
# elevation = ...
# residuals = ...
def elev_noise(elevation, residuals):
    plt.figure(figsize=(10, 6))  # 宽高适配PPT

    plt.scatter(elevation, residuals, s=10, c='tab:blue')  # 适当设置透明度和点大小

    plt.xlabel('Elevation [degree]', fontsize=14)
    plt.ylabel('Residuals [cycle]', fontsize=14)
    plt.title('Relation between phase noise and elevation', fontsize=16)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('elevation_vs_residuals.png', dpi=300)
    plt.show()

