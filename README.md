# Furniture Recommendation System

## How to Run

### Web Application

To run the web application:

```sh
turbo run dev --filter=web
```

### Server

To run the server:

```sh
python sever.py
```

## Project Structure

This project is organized as a Turborepo with multiple packages and apps:

### Apps and Packages

- **docs**: a Next.js app with Tailwind CSS for documentation purposes.
- **web**: another Next.js app with Tailwind CSS for the main user interface.
- **server**: a Flask server for handling image uploads and providing recommendations.
- **ui**: a React component library with Tailwind CSS shared by both web and docs applications.
- **@repo/eslint-config**: eslint configurations including eslint-config-next and eslint-config-prettier.
- **@repo/typescript-config**: TypeScript configurations used throughout the monorepo.

Each package/app is written in TypeScript for static type checking.

### Building UI Components

The `ui` package contains reusable React components styled with Tailwind CSS. These components are consumed by both the `web` and `docs` applications. 

The Tailwind CSS styles for the `ui` package are compiled into the `dist` directory for easy consumption. The component `.tsx` files are transpiled by the Next.js Compiler and `tailwindcss`. This approach ensures clear package export boundaries and prevents class name conflicts.

## Technologies Used

- **Python**: Backend development is done using Flask.
- **JavaScript/TypeScript**: Frontend development is done using Next.js and React.
- **Tailwind CSS**: Used for styling the user interface components.
- **TensorFlow**: Utilized for feature extraction and image classification in the recommendation system.
- **Flask**: Backend server framework for handling image uploads and providing recommendations.

## Usage

1. **Upload Image**: Users can upload an image of a furniture item they like.
2. **View Recommendations**: The system will display recommendations of similar furniture items based on the uploaded image.
3. **API Integration**: The system provides APIs for integrating the recommendation system into other applications.

