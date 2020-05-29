# Dockerfile
# without an file extension

FROM centos:7
RUN yum -y install gcc
RUN yum -y install make
RUN yum -y install wget
RUN yum -y install file

#FFTW
RUN wget http://www.fftw.org/fftw-3.3.8.tar.gz && tar xzf fftw-3.3.8.tar.gz
WORKDIR fftw-3.3.8
RUN ./configure --enable-shared --prefix=/usr/local/ && make && make install
WORKDIR ..

# Now Python packages
FROM centos/python-36-centos7

# Install numpy first for fdint in requirements
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install matplotlib

# Install any needed packages specified in requirements.txt 
COPY Docker/requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt 
RUN rm requirements.txt

# Install Sarkas
WORKDIR sarkas
COPY ./ sarkas/
