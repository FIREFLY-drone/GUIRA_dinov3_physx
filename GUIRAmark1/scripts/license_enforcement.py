#!/usr/bin/env python3
"""
🔒 Automated License Enforcement System
Fire Prevention System Repository

This script provides comprehensive license violation detection,
contributor management, and automated legal compliance enforcement.

Author: [Your Full Legal Name]
License: Proprietary - All Rights Reserved
Version: 2.0
"""

import json
import os
import re
import smtplib
import time
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
import sqlite3
from github import Github
from dataclasses import dataclass
from jinja2 import Template


@dataclass
class Contributor:
    """Contributor information and CLA status."""
    github_username: str
    legal_name: str
    email: str
    cla_signed_date: datetime
    cla_id: str
    status: str  # 'pending', 'approved', 'rejected', 'expired'
    employer: Optional[str] = None
    witness_name: Optional[str] = None


@dataclass
class ViolationAlert:
    """License violation detection result."""
    violation_type: str
    detected_url: str
    confidence_score: float
    description: str
    evidence: Dict
    timestamp: datetime


class LicenseEnforcementSystem:
    """Comprehensive license enforcement and contributor management."""
    
    def __init__(self, config_path: str = "enforcement_config.json"):
        """Initialize enforcement system."""
        self.config = self._load_config(config_path)
        self.github = Github(self.config["github_token"])
        self.repo = self.github.get_repo(self.config["repository"])
        self.db_path = self.config.get("database_path", "contributors.db")
        self._init_database()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        default_config = {
            "repository": "your-username/FIREPREVENTION",
            "owner_name": "[Your Full Legal Name]",
            "owner_email": "[your-email@domain.com]",
            "legal_email": "[legal-email@domain.com]",
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "violation_check_interval": 3600,  # 1 hour
            "cla_expiry_days": 365,
            "auto_block_violations": True
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return {**default_config, **config}
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using defaults")
            return default_config
    
    def _init_database(self):
        """Initialize SQLite database for contributor tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Contributors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contributors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                github_username TEXT UNIQUE NOT NULL,
                legal_name TEXT NOT NULL,
                email TEXT NOT NULL,
                cla_signed_date TIMESTAMP NOT NULL,
                cla_id TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL,
                employer TEXT,
                witness_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Violations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                violation_type TEXT NOT NULL,
                detected_url TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                description TEXT NOT NULL,
                evidence TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP
            )
        ''')
        
        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                user_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def process_cla_submission(self, contributor_data: Dict) -> str:
        """Process new CLA submission."""
        try:
            # Generate unique CLA ID
            cla_id = f"CLA-{datetime.now().strftime('%Y%m%d')}-{contributor_data['github_username'][:8].upper()}"
            
            # Create contributor record
            contributor = Contributor(
                github_username=contributor_data['github_username'],
                legal_name=contributor_data['legal_name'],
                email=contributor_data['email'],
                cla_signed_date=datetime.now(),
                cla_id=cla_id,
                status='pending',
                employer=contributor_data.get('employer'),
                witness_name=contributor_data.get('witness_name')
            )
            
            # Store in database
            self._store_contributor(contributor)
            
            # Send confirmation email
            self._send_cla_confirmation_email(contributor)
            
            # Log action
            self._log_action("cla_submitted", f"CLA submitted by {contributor.github_username}", contributor.github_username)
            
            return cla_id
            
        except Exception as e:
            self._log_action("cla_submission_error", f"Error processing CLA: {str(e)}", contributor_data.get('github_username'))
            raise
    
    def approve_contributor(self, cla_id: str, admin_notes: str = "") -> bool:
        """Approve contributor CLA."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update status to approved
            cursor.execute('''
                UPDATE contributors 
                SET status = 'approved', updated_at = CURRENT_TIMESTAMP 
                WHERE cla_id = ?
            ''', (cla_id,))
            
            if cursor.rowcount == 0:
                return False
            
            # Get contributor details
            cursor.execute('SELECT * FROM contributors WHERE cla_id = ?', (cla_id,))
            contributor_data = cursor.fetchone()
            
            conn.commit()
            conn.close()
            
            if contributor_data:
                # Send approval email
                self._send_approval_email(contributor_data)
                
                # Update GitHub repository secrets
                self._update_approved_contributors_secret()
                
                # Log action
                self._log_action("contributor_approved", f"CLA {cla_id} approved", contributor_data[1])
                
            return True
            
        except Exception as e:
            self._log_action("approval_error", f"Error approving CLA {cla_id}: {str(e)}")
            return False
    
    def check_license_violations(self) -> List[ViolationAlert]:
        """Comprehensive license violation detection."""
        violations = []
        
        # Check GitHub for similar repositories
        violations.extend(self._check_github_similarities())
        
        # Check for commercial use indicators
        violations.extend(self._check_commercial_usage())
        
        # Check patent databases
        violations.extend(self._check_patent_filings())
        
        # Check academic papers
        violations.extend(self._check_academic_citations())
        
        # Store violations in database
        for violation in violations:
            self._store_violation(violation)
        
        return violations
    
    def _check_github_similarities(self) -> List[ViolationAlert]:
        """Check GitHub for repositories with similar code."""
        violations = []
        
        # Search for repositories with similar names or descriptions
        search_terms = [
            "fire detection yolo",
            "drone fire monitoring",
            "wildfire ai detection",
            "forest fire computer vision",
            "smoke detection timesformer"
        ]
        
        for term in search_terms:
            try:
                repos = self.github.search_repositories(query=term, sort="updated")
                
                for repo in repos[:10]:  # Check top 10 results
                    if repo.full_name == self.config["repository"]:
                        continue
                    
                    # Analyze repository for potential violations
                    similarity_score = self._analyze_repo_similarity(repo)
                    
                    if similarity_score > 0.7:  # High similarity threshold
                        violation = ViolationAlert(
                            violation_type="code_similarity",
                            detected_url=repo.html_url,
                            confidence_score=similarity_score,
                            description=f"Repository {repo.full_name} shows high similarity to proprietary code",
                            evidence={
                                "repo_name": repo.full_name,
                                "description": repo.description,
                                "language": repo.language,
                                "stars": repo.stargazers_count,
                                "forks": repo.forks_count
                            },
                            timestamp=datetime.now()
                        )
                        violations.append(violation)
                        
            except Exception as e:
                print(f"Error checking GitHub similarities for '{term}': {e}")
                
        return violations
    
    def _check_commercial_usage(self) -> List[ViolationAlert]:
        """Check for commercial usage indicators."""
        violations = []
        
        # Search for commercial websites mentioning the technology
        commercial_indicators = [
            "fire detection service",
            "wildfire monitoring solution",
            "ai fire prevention consulting",
            "drone fire detection product"
        ]
        
        for indicator in commercial_indicators:
            # This would integrate with web scraping or search APIs
            # Placeholder for comprehensive commercial detection
            pass
            
        return violations
    
    def _check_patent_filings(self) -> List[ViolationAlert]:
        """Check patent databases for potential violations."""
        violations = []
        
        # This would integrate with patent database APIs
        # such as USPTO, EPO, WIPO
        # Placeholder for patent monitoring
        
        return violations
    
    def _check_academic_citations(self) -> List[ViolationAlert]:
        """Check academic papers for proper attribution."""
        violations = []
        
        # This would integrate with academic search APIs
        # such as Google Scholar, arXiv, IEEE Xplore
        # Placeholder for academic monitoring
        
        return violations
    
    def _analyze_repo_similarity(self, repo) -> float:
        """Analyze repository similarity to detect potential violations."""
        similarity_score = 0.0
        
        try:
            # Check README content
            try:
                readme = repo.get_readme()
                readme_content = readme.decoded_content.decode('utf-8').lower()
                
                # Look for specific terms from our project
                fire_terms = ["yolov8", "timesformer", "resnet50", "vari", "csrnet"]
                matches = sum(1 for term in fire_terms if term in readme_content)
                similarity_score += matches * 0.1
                
            except:
                pass
            
            # Check file structure
            try:
                contents = repo.get_contents("")
                file_names = [item.name.lower() for item in contents]
                
                # Look for similar file patterns
                our_files = ["train_fire.py", "train_smoke.py", "train_veg.py", "simulate_spread.py"]
                matches = sum(1 for our_file in our_files if our_file in file_names)
                similarity_score += matches * 0.15
                
            except:
                pass
            
            # Check repository description
            if repo.description:
                desc_lower = repo.description.lower()
                key_phrases = ["fire detection", "wildfire", "drone monitoring", "computer vision"]
                matches = sum(1 for phrase in key_phrases if phrase in desc_lower)
                similarity_score += matches * 0.1
                
        except Exception as e:
            print(f"Error analyzing repository {repo.full_name}: {e}")
            
        return min(similarity_score, 1.0)
    
    def _store_contributor(self, contributor: Contributor):
        """Store contributor in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO contributors 
            (github_username, legal_name, email, cla_signed_date, cla_id, status, employer, witness_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            contributor.github_username,
            contributor.legal_name,
            contributor.email,
            contributor.cla_signed_date,
            contributor.cla_id,
            contributor.status,
            contributor.employer,
            contributor.witness_name
        ))
        
        conn.commit()
        conn.close()
    
    def _store_violation(self, violation: ViolationAlert):
        """Store violation alert in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO violations 
            (violation_type, detected_url, confidence_score, description, evidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            violation.violation_type,
            violation.detected_url,
            violation.confidence_score,
            violation.description,
            json.dumps(violation.evidence)
        ))
        
        conn.commit()
        conn.close()
    
    def _log_action(self, action: str, details: str, user_id: str = None):
        """Log action to audit trail."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_log (action, details, user_id)
            VALUES (?, ?, ?)
        ''', (action, details, user_id))
        
        conn.commit()
        conn.close()
    
    def _send_cla_confirmation_email(self, contributor: Contributor):
        """Send CLA confirmation email."""
        template = Template('''
        <h2>🔒 CLA Submission Received - Fire Prevention System</h2>
        
        <p>Dear {{ contributor.legal_name }},</p>
        
        <p>We have received your signed Contributor License Agreement for the Fire Prevention System repository.</p>
        
        <h3>📋 Submission Details:</h3>
        <ul>
            <li><strong>CLA ID:</strong> {{ contributor.cla_id }}</li>
            <li><strong>GitHub Username:</strong> {{ contributor.github_username }}</li>
            <li><strong>Submission Date:</strong> {{ contributor.cla_signed_date.strftime('%Y-%m-%d %H:%M:%S UTC') }}</li>
            <li><strong>Status:</strong> Pending Review</li>
        </ul>
        
        <h3>⏳ Next Steps:</h3>
        <ol>
            <li>Legal review of your CLA (2-5 business days)</li>
            <li>Verification of witness signature</li>
            <li>Background check (if applicable)</li>
            <li>Final approval notification</li>
        </ol>
        
        <h3>⚖️ Legal Reminder:</h3>
        <p>Upon approval, all your contributions to the Fire Prevention System will become the exclusive property of {{ config.owner_name }}. You will retain no ownership rights and receive no compensation.</p>
        
        <h3>📞 Contact:</h3>
        <p>Questions? Contact {{ config.legal_email }}</p>
        
        <hr>
        <p><em>Fire Prevention System - Proprietary License Enforcement System</em></p>
        ''')
        
        html_content = template.render(contributor=contributor, config=self.config)
        
        self._send_email(
            to_email=contributor.email,
            subject=f"🔒 CLA Received - {contributor.cla_id}",
            html_content=html_content
        )
    
    def _send_approval_email(self, contributor_data: Tuple):
        """Send contributor approval email."""
        template = Template('''
        <h2>✅ CLA Approved - Welcome to Fire Prevention System</h2>
        
        <p>Dear {{ legal_name }},</p>
        
        <p>Congratulations! Your Contributor License Agreement has been <strong>approved</strong>.</p>
        
        <h3>📋 Approval Details:</h3>
        <ul>
            <li><strong>CLA ID:</strong> {{ cla_id }}</li>
            <li><strong>GitHub Username:</strong> {{ github_username }}</li>
            <li><strong>Approval Date:</strong> {{ approval_date }}</li>
            <li><strong>Status:</strong> ✅ APPROVED</li>
        </ul>
        
        <h3>🚀 You Can Now:</h3>
        <ul>
            <li>Submit pull requests to the repository</li>
            <li>Open issues and participate in discussions</li>
            <li>Contribute code, documentation, and tests</li>
            <li>Collaborate with the development team</li>
        </ul>
        
        <h3>📖 Contributing Guidelines:</h3>
        <p>Please review <a href="https://github.com/{{ repo_name }}/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a> for technical standards and development workflow.</p>
        
        <h3>⚖️ Legal Obligations:</h3>
        <ul>
            <li>All contributions become property of {{ owner_name }}</li>
            <li>No compensation or attribution guaranteed</li>
            <li>Commercial use rights assigned to owner</li>
            <li>Agreement is irrevocable and permanent</li>
        </ul>
        
        <h3>🛠️ Technical Resources:</h3>
        <ul>
            <li><a href="https://github.com/{{ repo_name }}/blob/main/TECHNICAL_ALGORITHMS_GUIDE.md">Technical Guide</a></li>
            <li><a href="https://github.com/{{ repo_name }}/blob/main/README.md">Project Overview</a></li>
            <li><a href="https://github.com/{{ repo_name }}/issues">Issue Tracker</a></li>
        </ul>
        
        <p>Welcome to the team! We look forward to your contributions.</p>
        
        <hr>
        <p><em>Fire Prevention System Development Team</em></p>
        ''')
        
        html_content = template.render(
            legal_name=contributor_data[2],
            cla_id=contributor_data[5],
            github_username=contributor_data[1],
            approval_date=datetime.now().strftime('%Y-%m-%d'),
            repo_name=self.config["repository"],
            owner_name=self.config["owner_name"]
        )
        
        self._send_email(
            to_email=contributor_data[3],
            subject=f"✅ CLA Approved - Welcome to Fire Prevention System",
            html_content=html_content
        )
    
    def _send_email(self, to_email: str, subject: str, html_content: str):
        """Send email notification."""
        try:
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config["owner_email"]
            msg['To'] = to_email
            
            html_part = MimeText(html_content, 'html')
            msg.attach(html_part)
            
            with smtplib.SMTP(self.config["smtp_server"], self.config["smtp_port"]) as server:
                server.starttls()
                server.login(self.config["smtp_username"], self.config["smtp_password"])
                server.send_message(msg)
                
            self._log_action("email_sent", f"Email sent to {to_email}: {subject}")
            
        except Exception as e:
            self._log_action("email_error", f"Failed to send email to {to_email}: {str(e)}")
    
    def _update_approved_contributors_secret(self):
        """Update GitHub repository secret with approved contributors."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT github_username FROM contributors WHERE status = "approved"')
            approved_usernames = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            # Update GitHub repository secret
            # This would require additional GitHub API setup for secrets management
            # Placeholder for secret update logic
            
            self._log_action("secret_updated", f"Updated approved contributors list: {len(approved_usernames)} contributors")
            
        except Exception as e:
            self._log_action("secret_update_error", f"Failed to update secrets: {str(e)}")
    
    def run_continuous_monitoring(self):
        """Run continuous license monitoring."""
        print("🔒 Starting License Enforcement System...")
        print(f"Repository: {self.config['repository']}")
        print(f"Owner: {self.config['owner_name']}")
        print(f"Check interval: {self.config['violation_check_interval']} seconds")
        
        while True:
            try:
                print(f"\n🔍 Running violation check at {datetime.now()}")
                
                # Check for violations
                violations = self.check_license_violations()
                
                if violations:
                    print(f"⚠️ Found {len(violations)} potential violations")
                    for violation in violations:
                        print(f"  - {violation.violation_type}: {violation.detected_url} (confidence: {violation.confidence_score:.2f})")
                        
                        # Send immediate alert for high-confidence violations
                        if violation.confidence_score > 0.8:
                            self._send_violation_alert(violation)
                else:
                    print("✅ No violations detected")
                
                # Check for expired CLAs
                self._check_cla_expiry()
                
                # Sleep until next check
                time.sleep(self.config["violation_check_interval"])
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping License Enforcement System...")
                break
            except Exception as e:
                print(f"❌ Error during monitoring: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _check_cla_expiry(self):
        """Check for expiring CLAs."""
        if self.config.get("cla_expiry_days", 0) <= 0:
            return  # No expiry configured
        
        expiry_date = datetime.now() - timedelta(days=self.config["cla_expiry_days"])
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT github_username, email, cla_signed_date 
            FROM contributors 
            WHERE status = "approved" AND cla_signed_date < ?
        ''', (expiry_date,))
        
        expiring_contributors = cursor.fetchall()
        
        for contributor in expiring_contributors:
            print(f"⚠️ CLA expired for {contributor[0]}")
            # Send renewal notice
            self._send_cla_renewal_notice(contributor)
        
        conn.close()
    
    def _send_violation_alert(self, violation: ViolationAlert):
        """Send immediate violation alert."""
        template = Template('''
        <h2>🚨 URGENT: License Violation Detected</h2>
        
        <h3>⚠️ Violation Details:</h3>
        <ul>
            <li><strong>Type:</strong> {{ violation.violation_type.title() }}</li>
            <li><strong>URL:</strong> <a href="{{ violation.detected_url }}">{{ violation.detected_url }}</a></li>
            <li><strong>Confidence:</strong> {{ (violation.confidence_score * 100)|round(1) }}%</li>
            <li><strong>Detected:</strong> {{ violation.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') }}</li>
        </ul>
        
        <h3>📝 Description:</h3>
        <p>{{ violation.description }}</p>
        
        <h3>🔍 Evidence:</h3>
        <pre>{{ violation.evidence | tojson(indent=2) }}</pre>
        
        <h3>⚡ Recommended Actions:</h3>
        <ol>
            <li>Review the detected violation immediately</li>
            <li>Document evidence for legal proceedings</li>
            <li>Contact legal counsel if necessary</li>
            <li>Consider sending cease and desist notice</li>
        </ol>
        
        <hr>
        <p><em>Automated License Enforcement System - Fire Prevention Project</em></p>
        ''')
        
        html_content = template.render(violation=violation)
        
        self._send_email(
            to_email=self.config["legal_email"],
            subject=f"🚨 URGENT: License Violation Detected - {violation.violation_type}",
            html_content=html_content
        )


def main():
    """Main entry point for license enforcement system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fire Prevention System License Enforcement")
    parser.add_argument("--config", default="enforcement_config.json", help="Configuration file path")
    parser.add_argument("--monitor", action="store_true", help="Run continuous monitoring")
    parser.add_argument("--check-violations", action="store_true", help="Run one-time violation check")
    parser.add_argument("--approve-cla", help="Approve CLA by ID")
    
    args = parser.parse_args()
    
    # Initialize system
    enforcement = LicenseEnforcementSystem(args.config)
    
    if args.monitor:
        enforcement.run_continuous_monitoring()
    elif args.check_violations:
        violations = enforcement.check_license_violations()
        print(f"Found {len(violations)} potential violations")
        for v in violations:
            print(f"  {v.violation_type}: {v.detected_url} ({v.confidence_score:.2f})")
    elif args.approve_cla:
        success = enforcement.approve_contributor(args.approve_cla)
        print(f"CLA approval {'successful' if success else 'failed'}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
