# 1. Vllm 部署
在自己电脑上部署LLM Model，建议直接使用Ollama就行，Ollama对内存的占用相对较少，适合资源受限的设备，比如笔记本或者一台台式机。

但是如果你想部署一个LLM Model，供整个课题组或者整个学院（也就是并发量大的情况）来使用的话，Ollama是受不了的，它的并发处理能力相对有限。这种情况下，就得考虑 `vllm`。

```python
pip install vllm
```

<img width="869" height="504" alt="image" src="https://github.com/user-attachments/assets/da8bd3a3-ccb1-4098-9ced-45100f239b93" />

以 `Qwen3-14B` 为例，可以使用 `modelscope download --model Qwen/Qwen3-14B` 命令下载到本地，一般会存储在 `/home/user/.cache/modelscope/hub/models/Qwen`，也可以下载到指定目录进行下载，只需要在终端运行:
```python
  export MODELSCOPE_CACHE=/your/target/path modelscope download --model Qwen/Qwen3-14B
```
<img width="1440" height="805" alt="image" src="https://github.com/user-attachments/assets/ebf67a3e-2315-4b5f-817b-f5898e55133f" />
> Qwen3-14B系列的模型需要4张A6000或者2张A100的卡。  

\
下载完成后，按照以下命令执行：  \
1. `export CUDA_VISIBLE_DEVICES=0,1,2,3`  <br/>
2. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` # 配置 PyTorch 的 CUDA 显存分配行为，用来减少显存碎片，提高内存利用率。\
3. `export NCCL_P2P_DISABLE=1` # 禁用 NCCL 的 P2P 功能，强制使用 主机内存中转（通过 CPU + PCIe）进行跨 GPU 通信 \
4. `python -m vllm.entrypoints.openai.api_server --model /home/user/workspace/LLM/models/Qwen/Qwen3-14B --trust-remote-code --tensor-parallel-size 4 --max-model-len 8192` # --model切换成自己的模型所在路

若需要关闭思考模式，执行:
```python
python -m vllm.entrypoints.openai.api_server --model /home/user/workspace/LLM/models/Qwen/Qwen3-14B --trust-remote-code --tensor-parallel-size 4 --max-model-len 8192 --enable-reasoning --reasoning-parser deepseek_r1
```

出现以下界面，代码部署成功;
<img width="1440" height="453" alt="image" src="https://github.com/user-attachments/assets/41b7c029-05b4-4e3b-9068-fbf07105a3c0" />

模型测试 (XX.XX.XXX.XX换成自己部署服务器的 IP )：
```python
curl -X POST "http://XX.XX.XXX.XX:8000/v1/chat/completions" \   -H "Content-Type: application/json" \   -H "Authorization: Bearer EMPTY" \   -d '{     "model": "/home/user/workspace/LLM/models/Qwen/Qwen3-14B",     "messages": [{"role": "user", "content": "小学奥数值得学吗"}],     "stream": false }'
```

模型连接断开的话，按照以下命令重新连接 (先切换到对应vllm的conda环境)：
   1. `export CUDA_VISIBLE_DEVICES=0,1,2,3`
   2. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   3. `export NCCL_P2P_DISABLE=1`
   4. `python -m vllm.entrypoints.openai.api_server --model /home/user/workspace/LLM/models/Qwen/Qwen3-14B --trust-remote-code --tensor-parallel-size 4 --max-model-len 8192 --enable-reasoning --reasoning-parser deepseek_r1`


# 2. Systemd 服务
若是在一个集群上部署出现这样的问题，建议做成Linux服务，不然肯定会不太稳定的。这样做的意义在于无论是因为网络、开关机等问题导致的 vLLM 服务断开，可以通过创建一个 systemd 服务单元文件来实现在断开后可以自动重新启动vLLM连接LLM服务。
- 创建启动脚本，打开终端并创建启动脚本，命令如下（这里的命令要将 username换成你自己的用户名，Linux下一般都是user）：
  `vim /home/username/start_vllm.sh`
  <img width="639" height="365" alt="image" src="https://github.com/user-attachments/assets/0b8106c0-4eda-4c05-9742-d07f7dc108e5" />
- 在创建的 `start_vllm.sh` 文件中添加以下内容，将 vllm 替换为您的 conda 环境的名称，
  > vim命令执行会，会自动打开start_vllm.sh，这个时候你看到的是 Vim 的普通模式，无法直接输入文字。 需要按下键盘上的 i 进入插入模式才可进行编辑，此时左下角会显示 -- INSERT --。 现在就可以输入或粘贴你的服务配置内容了。
  ```
    #!/bin/bash
  # 加载 conda 环境
  source /home/user/anaconda3/etc/profile.d/conda.sh # 改为自己正确的路径
  conda activate vllm  # 改为自己vllm的环境名
  
  kill -9 $(nvidia-smi | grep python | awk '{print $5}')
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export NCCL_P2P_DISABLE=1
   
  # 启动 vllm 服务，下面的参数可按自己的需求进行设置，也可以在调用接口的时候进行设置
  python -m vllm.entrypoints.openai.api_server \
  --model /home/user/workspace/LLM/models/Qwen/Qwen3-14B \
  --trust-remote-code --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --enable-reasoning --reasoning-parser deepseek_r1
  ```
- 按 `esc` 键，然后在终端输入 `:wq`，再进行回车，以保存文件和退出，
- 修改权限，使脚本可执行（注意要将 username换成你自己的用户名）:
  `chmod +x /home/user/start_vllm.sh`
- 创建 vLLM 的 systemd 服务文件，
  > 在/etc/systemd/system目录下创建文件，一般需要权限的，需要输入密码（也就是ssh连接的密码）
  `sudo vim /etc/systemd/system/vllm.service`
- 在服务文件 vllm.service中添加以下内容，将 user 替换为您的实际用户名，
  ```
  [Unit]
  Description=VLLM Service for SQLCoder Model
  After=network.target
   
  [Service]
  Type=simple
  User=user
  WorkingDirectory=/home/user
  ExecStart=/bin/bash /home/user/start_vllm.sh
  Restart=always
   
  [Install]
  WantedBy=multi-user.target
  ```
- 按 `esc` 键，然后在终端输入:wq，再进行回车，以保存文件和退出
- 重新加载 systemd 并启用服务
  ```
  # 重新加载 systemd 服务
  sudo systemctl daemon-reload
  # 启动服务测试
  sudo systemctl start vllm.service
  # 启用服务以便开机启动
  sudo systemctl enable vllm.service
  ```
- 检查服务启动状态
  `sudo systemctl status vllm.service`
  <img width="1125" height="468" alt="image" src="https://github.com/user-attachments/assets/3c24696a-bab8-4469-9755-4c28042d51e1" />

出现 active（running） 表示启动成功。
