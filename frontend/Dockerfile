# Use Node 18 for Next.js
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package.json and install dependencies
COPY dlmodeldeployment/package.json ./
RUN npm install --force

# Copy all files to the container
COPY dlmodeldeployment/ ./

# Build the Next.js application
RUN npm run build

# Expose the desired port for the frontend
EXPOSE 3000

# Start the Next.js application
CMD ["npm", "start"]
