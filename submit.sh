#!/usr/bin/env bash
# Since the token expires real quick, prompt the user to get the token each time
# before submitting the Docker image.
set -eo pipefail

echo "Start up and enter the Vertex Cloud VM."
echo "Run 'gcloud auth print-access-token' to get the short-lived token."
echo "Input token:"
read -r token

# Upload Docker image.
echo $token | docker login -u oauth2accesstoken --password-stdin https://asia-southeast1-docker.pkg.dev
docker push asia-southeast1-docker.pkg.dev/til-ai-2025/h3althydr0plet-repo-til-25/h3althydr0plet-cv:latest

echo $token > /tmp/access_token.txt

# Submit image.
gcloud ai models upload \
  --access-token-file /tmp/access_token.txt \
  --project til-ai-2025 \
  --region asia-southeast1 \
  --display-name h3althydr0plet-cv \
  --container-image-uri asia-southeast1-docker.pkg.dev/til-ai-2025/h3althydr0plet-repo-til-25/h3althydr0plet-cv:latest \
  --container-health-route /health \
  --container-predict-route /cv \
  --container-ports 5002 \
  --version-aliases default
