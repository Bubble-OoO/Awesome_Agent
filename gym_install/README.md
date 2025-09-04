1. 创建新的conda环境
   
   ```python
    conda create -n gym_env python=3.10
    conda activate gym_env
   ```

2. 安装box2d-py
   
   ```python
    conda install -c conda-forge box2d-py
   ```

3. 安装其他依赖

   ```python
    conda install -c conda-forge pygame
    pip install gymnasium
   ```

4. 测试安装

   ```python
    python -c "import gymnasium as gym; env = gym.make('LunarLander-v3'); print('安装成功！')"
   ```
