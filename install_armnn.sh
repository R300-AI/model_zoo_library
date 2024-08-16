wget -O ArmNN-aarch64.tgz https://github.com/ARM-software/armnn/releases/download/v23.08/ArmNN-linux-aarch64.tar.gz
mkdir armnn
tar -xvf ArmNN-aarch64.tgz -C armnn

sudo ln ./armnn/libarmnnDelegate.so.29.0 libarmnnDelegate.so.29
sudo ln ./armnn/libarmnn.so.33.0 libarmnn.so.33
