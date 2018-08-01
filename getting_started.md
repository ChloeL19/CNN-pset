# Setting up in your Cloud9 workspace

Note: Make sure your workspace is resized to at least 5 GB of disk space and 1 GB of memory.
Total disk space taken up by these downloads: 3.1GB

### Installing miniconda3 (useful for installing later dependencies): 0.4GB
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod a+x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh

*then delete the .sh file: rm Miniconda3-latest-Linux-x86_64.sh

### Installing core dependencies: 2.7GB
conda install numpy 

conda install tensorflow

conda install keras 

conda install matplotlib 

conda install opencv 

### Download EMNIST dataset: 0.04GB (basically negligible)
wget http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip

unzip matlab.zip

rm matlab.zip

*delete all of the .mat files in the matlab folder except for the emnist-balanced.mat file

*rename the matlab folder to EMNIST_data (for clarity, and this way you won’t have to tweak the EMNIST staff upload code)

### Download your preferred distro code: (negligible size)
Code for less comfy students to complete: wget https://raw.githubusercontent.com/ChloeL19/CNN-pset/master/Student_Code/c9_less_comfy/less_comfy.zip

Code for more comfy students to complete: wget https://raw.githubusercontent.com/ChloeL19/CNN-pset/master/Student_Code/c9_more_comfy/student_more_comfy.zip

Example staff solution for less comfy folks: wget https://raw.githubusercontent.com/ChloeL19/CNN-pset/master/Staff_Code/c9_less_comfy/less_comfy_staff.zip

Example staff solution for more comfy folks:   wget https://raw.githubusercontent.com/ChloeL19/CNN-pset/master/Staff_Code/c9_more_comfy/more_comfy_staff.zip

unzip your zip file and then delete it

### Have fun!
