FROM hashicorp/terraform:1.1.2

RUN apk add --update --no-cache git curl bash unzip

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    sudo ./aws/install && \
    rm awscliv2.zip

# Install jq
RUN apk add --update --no-cache jq

# Set Terraform version
ENV TERRAFORM_VERSION=1.1.2

ENTRYPOINT [ "/bin/bash" ]
