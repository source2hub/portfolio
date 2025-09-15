# Overview

This is a full-stack web application featuring a modern data scientist portfolio built with React and TypeScript. The project follows a monorepo structure with a React frontend using shadcn/ui components, an Express.js backend with TypeScript, and PostgreSQL database integration via Drizzle ORM. The application is designed to showcase professional data science projects, skills, and provide contact functionality.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: React 18 with TypeScript and Vite as the build tool
- **UI Library**: shadcn/ui components built on Radix UI primitives
- **Styling**: Tailwind CSS with CSS variables for theming support
- **State Management**: TanStack React Query for server state management
- **Routing**: Wouter for lightweight client-side routing
- **Component Structure**: Modular component architecture with separate sections for hero, about, projects, skills, and contact

## Backend Architecture
- **Runtime**: Node.js with Express.js framework
- **Language**: TypeScript with ES modules
- **API Design**: RESTful API structure with /api prefix for all endpoints
- **Storage Interface**: Abstracted storage layer with in-memory implementation (MemStorage) and database interface (IStorage)
- **Development Setup**: Vite integration for development with HMR support

## Data Storage Solutions
- **Database**: PostgreSQL with Drizzle ORM for type-safe database operations
- **Connection**: Neon Database serverless driver for PostgreSQL connectivity
- **Schema Management**: Drizzle Kit for database migrations and schema management
- **Session Storage**: PostgreSQL-based session storage using connect-pg-simple

## Authentication and Authorization
- **Session Management**: Express sessions with PostgreSQL session store
- **User Schema**: Basic user model with username/password authentication
- **Security**: Prepared for authentication implementation with user creation and retrieval methods

## External Dependencies
- **Database Provider**: Neon Database (serverless PostgreSQL)
- **UI Components**: Radix UI primitives for accessible component foundation
- **Icons**: Lucide React icons and React Icons (Simple Icons)
- **Fonts**: Google Fonts integration (Inter, Architects Daughter, DM Sans, Fira Code, Geist Mono)
- **Build Tools**: Vite with React plugin and TypeScript support
- **Development Tools**: Replit-specific plugins for error overlay, cartographer, and dev banner

## Design Patterns
- **Component Composition**: Leverages Radix UI's composable component patterns
- **Type Safety**: Full TypeScript implementation with strict type checking
- **Separation of Concerns**: Clear separation between client, server, and shared code
- **Environment Configuration**: Environment-based configuration for database connections
- **Asset Management**: Centralized asset handling through Vite's alias system