FROM tensorflow/tensorflow:latest-gpu
# copy certificate
COPY key /root/key

# install code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh && \
mkdir -p /root/.config/code-server && \
echo 'bind-addr: 127.0.0.1:8080' >> /root/.config/code-server/config.yaml && \
echo 'auth: password' >> /root/.config/code-server/config.yaml && \
echo 'password: 47c0ca0f78b3c6196f13e726' >> /root/.config/code-server/config.yaml && \
echo 'cert: false' >> /root/.config/code-server/config.yaml && \
pip3 install progressbar2==3.47.0 pyworld==0.2.8 librosa==0.6.3 pylint


CMD code-server --cert /root/key/Nginx/1_www.xe-p1.top_bundle.crt --cert-key /root/key/Nginx/2_www.xe-p1.top.key  --bind-addr 0.0.0.0:8080 --user-data-dir /root/.vscode-server