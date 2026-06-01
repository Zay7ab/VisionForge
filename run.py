"""
VisionForge - Docker Launcher & Orchestrator

Automated deployment script that:
    1. Validates Docker installation
    2. Builds Docker images for backend and frontend
    3. Starts containers with proper configuration
    4. Opens browser to the application
    5. Streams logs for monitoring

Usage:
    python run.py

Requirements:
    - Docker Desktop installed and running
    - Docker Compose installed (usually bundled with Docker Desktop)

Troubleshooting:
    - If Docker is not found, install from: https://docs.docker.com/get-docker/
    - If containers fail to start, check docker logs: docker compose logs
    - To stop: Ctrl+C or in another terminal: docker compose down

Author: VisionForge Team
Version: 4.0
"""

import subprocess
import sys
import os
import time
import webbrowser

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def run_command(cmd, cwd=None):
    """
    Execute a shell command and return the process handle.
    
    Args:
        cmd: Shell command string
        cwd: Working directory for execution
        
    Returns:
        subprocess.Popen object representing the running process
    """
    return subprocess.Popen(cmd, shell=True, cwd=cwd)

def check_docker():
    """
    Verify Docker and Docker Compose are installed and available.
    
    Checks:
        1. Docker CLI exists (docker --version)
        2. Docker Compose CLI exists (docker compose --version)
    
    Returns:
        bool: True if both are available, False otherwise
    """
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        subprocess.run(["docker", "compose", "--version"], check=True, capture_output=True)
        return True
    except:
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main orchestration function:
        1. Display welcome banner
        2. Check Docker dependencies
        3. Build images from Dockerfiles
        4. Start containers with docker-compose
        5. Open browser and stream logs
    """
    # Display welcome banner
    print("=" * 60)
    print("   🧠  VisionForge - AI Image Classifier")
    print("=" * 60)
    
    # Validate Docker installation
    if not check_docker():
        print("\n❌ Docker is not installed or not running!")
        print("   Please install Docker and Docker Compose first.")
        print("   Visit: https://docs.docker.com/get-docker/")
        sys.exit(1)
    
    print("\n  🐳 Building and starting containers...")
    print("  This may take a few minutes on first run...")
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Step 1: Build Docker images from Dockerfiles
        # This creates:
        #   - teachable-backend: FastAPI server from backend/Dockerfile
        #   - teachable-frontend: Streamlit app from frontend/Dockerfile
        subprocess.run(["docker", "compose", "build"], check=True)
        
        # Step 2: Start containers in background mode (-d flag)
        # This runs:
        #   - Backend on http://localhost:8000
        #   - Frontend on http://localhost:8501
        subprocess.run(["docker", "compose", "up", "-d"], check=True)
        
        # Step 3: Wait for services to fully initialize
        print("\n  ⏳ Waiting for services to start...")
        time.sleep(5)
        
        # Step 4: Open browser to frontend
        print("  🌐 Opening browser...")
        webbrowser.open("http://localhost:8501")
        
        # Display service URLs and helpful information
        print("\n" + "=" * 60)
        print("  ✅  App is running!")
        print("  📱 Frontend: http://localhost:8501")
        print("  🔧 Backend:  http://localhost:8000")
        print("  📚 API Docs: http://localhost:8000/docs")
        print("=" * 60)
        print("\n  💡 Commands:")
        print("     - View logs:    docker compose logs -f")
        print("     - Stop:         docker compose down")
        print("     - Restart:      docker compose restart")
        print("     - Rebuild:      docker compose up --build")
        print("\n  Press Ctrl+C to stop viewing logs...")
        
        # Step 5: Stream container logs to console for monitoring
        # This provides real-time visibility into what's happening
        subprocess.run(["docker", "compose", "logs", "-f"])
        
    except KeyboardInterrupt:
        # Handle user interrupt (Ctrl+C)
        print("\n\n  🛑 Shutting down...")
    except Exception as e:
        # Handle unexpected errors
        print(f"\n  ❌ Error: {e}")
    finally:
        # Cleanup: Stop all containers gracefully
        print("  📦 Stopping containers...")
        subprocess.run(["docker", "compose", "down"])
        print("  ✅ Done!")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()