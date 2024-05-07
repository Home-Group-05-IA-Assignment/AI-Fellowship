# Use the specified base image
FROM 32bitbradley/cimg-python310-ansible

# Set the working directory inside the container
WORKDIR /home/circleci/project

# Set the environment variable for the Google API key
ENV GOOGLE_API_KEY=AIzaSyCAyZdZDa85vLQq-ZA-MNi5XWaBP0UW3AQ

# Update pip to the latest version
RUN pip install --upgrade pip

# Copy the requirements.txt file into the container
COPY . .

# Install Python dependencies from the requirements.txt file
RUN pip install -r requirements.txt