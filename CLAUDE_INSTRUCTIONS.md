# Claude Instructions for FinGuard

## Project Overview
This document defines how Claude should work with the FinGuard codebase.

## Core Principles

### 1. Code Quality Standards
- Write clean, maintainable, and well-documented code
- Follow language-specific best practices and conventions
- Prioritize readability over cleverness
- Keep functions small and focused on a single responsibility

### 2. Testing Requirements
- Write tests for all new features
- Maintain or improve existing test coverage
- Include unit tests, integration tests where appropriate
- Test edge cases and error handling

### 3. Documentation Standards
- Document all public APIs and functions
- Keep README.md up to date with setup and usage instructions
- Add inline comments only when logic isn't self-evident
- Document architectural decisions in code comments or separate docs

### 4. Git Workflow
- Write clear, descriptive commit messages
- Keep commits atomic and focused
- Follow conventional commit format when applicable
- Never commit sensitive data or credentials

### 5. Security Practices
- Validate all user inputs
- Sanitize data to prevent injection attacks
- Keep dependencies up to date
- Never expose sensitive information in logs or errors
- Follow OWASP security guidelines

### 6. Architecture Patterns
- Maintain separation of concerns
- Use dependency injection where appropriate
- Keep business logic separate from infrastructure code
- Follow existing architectural patterns in the codebase

### 7. Performance Considerations
- Optimize for readability first, performance second
- Profile before optimizing
- Avoid premature optimization
- Consider scalability in design decisions

## Task Workflow

### When Starting a New Task:
1. Read and understand existing relevant code
2. Create a todo list for complex tasks
3. Ask clarifying questions if requirements are unclear
4. Plan the approach before implementing

### During Implementation:
1. Make incremental changes
2. Test as you go
3. Update todos to track progress
4. Commit logical chunks of work

### Before Completing:
1. Review code for quality and security
2. Ensure all tests pass
3. Update documentation
4. Mark all todos as completed

## Technology Stack
(To be defined as the project evolves)

## Additional Notes
- Always prioritize maintainability over quick fixes
- When in doubt, ask for clarification
- Keep the user informed of progress on complex tasks
- Follow the principle: "Make it work, make it right, make it fast" - in that order
