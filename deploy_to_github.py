#!/usr/bin/env python3
"""
GitHub Deployment Script for IC Light Professional
Automates the process of pushing to GitHub for easy Colab access
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and print the result"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False
    return True


def check_git_installed():
    """Check if Git is installed"""
    try:
        result = subprocess.run("git --version", shell=True, capture_output=True)
        return result.returncode == 0
    except:
        return False


def main():
    """Main deployment function"""
    print("🚀 IC Light Professional - GitHub Deployment")
    print("=" * 50)
    
    # Check if Git is installed
    if not check_git_installed():
        print("❌ Git is not installed. Please install Git first.")
        print("Download from: https://git-scm.com/downloads")
        return
    
    print("✅ Git is available")
    
    # Get repository details
    print("\n📝 Repository Configuration:")
    repo_name = input("Enter repository name (default: ic-light-professional): ").strip()
    if not repo_name:
        repo_name = "ic-light-professional"
    
    username = input("Enter your GitHub username: ").strip()
    if not username:
        print("❌ GitHub username is required")
        return
    
    # Initialize Git repository
    commands = [
        ("git init", "Initializing Git repository"),
        ("git add .", "Adding all files to Git"),
        ("git commit -m 'Initial commit: IC Light Professional v1.0.0'", "Creating initial commit"),
        (f"git branch -M main", "Setting main branch"),
        (f"git remote add origin https://github.com/{username}/{repo_name}.git", "Adding GitHub remote")
    ]
    
    print(f"\n🔧 Setting up repository '{repo_name}' for user '{username}'...")
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"\n⚠️ Command failed: {command}")
            print("You may need to:")
            print("1. Create the repository on GitHub first")
            print("2. Configure your Git credentials")
            print("3. Check if the repository already exists")
            return
    
    # Push to GitHub
    print(f"\n🚀 Pushing to GitHub...")
    if run_command("git push -u origin main", "Pushing to GitHub"):
        print(f"\n🎉 Successfully deployed to GitHub!")
        print(f"📱 Repository URL: https://github.com/{username}/{repo_name}")
        print(f"🔗 Colab URL: https://colab.research.google.com/github/{username}/{repo_name}/blob/main/IC_Light_Professional_Colab.ipynb")
        print("\n📋 Next Steps:")
        print("1. Open the Colab URL above")
        print("2. Run all cells in the notebook")
        print("3. Start creating amazing lighting effects!")
        
        # Update the notebook with correct repository URL
        update_notebook_url(username, repo_name)
        
    else:
        print("\n❌ Failed to push to GitHub")
        print("Please check:")
        print("1. Repository exists on GitHub")
        print("2. You have push permissions")
        print("3. Your Git credentials are configured")
        print("\nManual steps:")
        print(f"1. Create repository: https://github.com/new")
        print(f"2. Repository name: {repo_name}")
        print("3. Make it public for Colab access")
        print("4. Run: git push -u origin main")


def update_notebook_url(username, repo_name):
    """Update notebook with correct repository URL"""
    try:
        notebook_path = Path(__file__).parent / "IC_Light_Professional_Colab.ipynb"
        if notebook_path.exists():
            # Read notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update GitHub URLs
            content = content.replace(
                "https://github.com/your-username/ic-light-professional.git",
                f"https://github.com/{username}/{repo_name}.git"
            )
            
            # Write back
            with open(notebook_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print("✅ Updated notebook with correct repository URL")
            
            # Commit the update
            run_command("git add IC_Light_Professional_Colab.ipynb", "Adding updated notebook")
            run_command("git commit -m 'Update notebook with correct repository URL'", "Committing URL update")
            run_command("git push", "Pushing URL update")
            
    except Exception as e:
        print(f"⚠️ Could not update notebook URL: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Deployment cancelled by user")
    except Exception as e:
        print(f"\n💥 Deployment failed: {e}")
        sys.exit(1)
