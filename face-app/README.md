# Manual downloads for project setup

## Follow the steps in order for the app to be manually setup correctly

1) ### Install docker
- Linux :
    ```bash
    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    
    # Add the repository to Apt sources:
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    ```
- Windows :`
    Install Docker Desktop for windows (<a href = https://docs.docker.com/desktop/setup/install/windows-install>Download link </a>)


2) ### Install Nvidia Docker
    ```bash
    # Add nvidia repo
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    # Update packages
    sudo apt-get update

    # Install Nvidia Docker
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
    ```

3) ### Download all external files not cloned from the repository (list below)

- The file **inswapper_128.onnx** shall be placed in the **/face-app/models/** folder. (<a href = "https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view?usp=sharing">Download link</a>)

- Put images from the celebA dataset in the project folder under the name **/data/img_align_celeba** (you may take less than all of the 200 000 files if it takes too much time) (<a href = "https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ">Download link</a>)


_Note:_ **ALL** other **models** and **libraries** are automatically fetched and downloaded from other repoitories and can take some time to complete.

---

# Docker setup for faster results and execution

4) ### Run the preprocessing docker-compose file with the command

```bash
docker-compose -f docker-compose-preprocess.yml up --build
```

#### **DO NOT MODIFY OR DELETE ANY FILE AFTER RUNNING THIS COMMAND**

5) ### After it returns 0, run the command

```bash
docker-compose -f docker-compose-preprocess.yml down
```

6) ### Run the app and wait for execution (process might take a while depending on the number of images copied in the preprocessing step).

```bash
docker-compose -f docker-compose-app.yml up --build
```

---
_Note_ : Results of best match should appear as **/data/output/best_match.jpg**
