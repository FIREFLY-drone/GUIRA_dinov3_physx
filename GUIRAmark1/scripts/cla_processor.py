#!/usr/bin/env python3
"""
✍️ CLA Processing and Management System
Fire Prevention System Repository

This script automates the processing of Contributor License Agreements,
manages contributor databases, and handles approval workflows.

Author: [Your Full Legal Name]
License: Proprietary - All Rights Reserved
Version: 2.0
"""

import argparse
import json
import os
import smtplib
import sqlite3
from datetime import datetime, timedelta
from email.mime.multipart import MimeMultipart
from email.mime.text import MimeText
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import secrets
import hashlib


@dataclass
class Contributor:
    """Contributor information and CLA status."""
    id: Optional[int] = None
    github_username: str = ""
    legal_name: str = ""
    email: str = ""
    cla_signed_date: Optional[datetime] = None
    cla_id: str = ""
    status: str = "pending"  # 'pending', 'approved', 'rejected', 'expired'
    employer: Optional[str] = None
    witness_name: Optional[str] = None
    document_hash: Optional[str] = None
    approval_date: Optional[datetime] = None
    notes: str = ""


class CLAProcessor:
    """Comprehensive CLA processing and management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize CLA processor."""
        self.config = self._load_config(config_path)
        self.db_path = self.config.get("database_path", "contributors.db")
        self._init_database()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or environment."""
        default_config = {
            "repository": "your-username/FIREPREVENTION",
            "owner_name": "[Your Full Legal Name]",
            "owner_email": "[your-email@domain.com]",
            "legal_email": "[legal-email@domain.com]",
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_username": os.getenv("SMTP_USERNAME"),
            "smtp_password": os.getenv("SMTP_PASSWORD"),
            "cla_expiry_days": 365,
            "require_witness": True,
            "require_notarization": False,
            "auto_approve_domains": [],  # Pre-approved email domains
            "blocked_domains": ["tempmail.com", "10minutemail.com"]
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                file_config = json.load(f)
            return {**default_config, **file_config}
        
        return default_config
    
    def _init_database(self):
        """Initialize SQLite database for contributor management."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Contributors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contributors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                github_username TEXT UNIQUE NOT NULL,
                legal_name TEXT NOT NULL,
                email TEXT NOT NULL,
                cla_signed_date TIMESTAMP,
                cla_id TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                employer TEXT,
                witness_name TEXT,
                document_hash TEXT,
                approval_date TIMESTAMP,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # CLA documents table (for storing document metadata)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cla_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contributor_id INTEGER,
                document_path TEXT,
                document_hash TEXT,
                file_size INTEGER,
                mime_type TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (contributor_id) REFERENCES contributors (id)
            )
        ''')
        
        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                contributor_id INTEGER,
                details TEXT,
                performed_by TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Email log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipient_email TEXT NOT NULL,
                subject TEXT NOT NULL,
                email_type TEXT NOT NULL,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def process_new_cla(self, contributor_data: Dict) -> Tuple[bool, str]:
        """Process a new CLA submission."""
        try:
            # Validate contributor data
            validation_result = self._validate_contributor_data(contributor_data)
            if not validation_result[0]:
                return False, validation_result[1]
            
            # Check for existing contributor
            existing = self._get_contributor_by_username(contributor_data['github_username'])
            if existing and existing.status == 'approved':
                return False, f"Contributor {contributor_data['github_username']} already has approved CLA"
            
            # Generate unique CLA ID
            cla_id = self._generate_cla_id(contributor_data['github_username'])
            
            # Create contributor record
            contributor = Contributor(
                github_username=contributor_data['github_username'],
                legal_name=contributor_data['legal_name'],
                email=contributor_data['email'],
                cla_signed_date=datetime.now(),
                cla_id=cla_id,
                status='pending',
                employer=contributor_data.get('employer'),
                witness_name=contributor_data.get('witness_name'),
                notes=contributor_data.get('notes', '')
            )
            
            # Store in database
            contributor_id = self._store_contributor(contributor)
            
            # Log action
            self._log_action("cla_submitted", contributor_id, 
                           f"CLA submitted by {contributor.github_username}")
            
            # Send confirmation email
            self._send_cla_confirmation_email(contributor)
            
            # Auto-approve if from trusted domain
            if self._should_auto_approve(contributor.email):
                self.approve_contributor(cla_id, "Auto-approved from trusted domain")
            
            return True, cla_id
            
        except Exception as e:
            self._log_action("cla_submission_error", None, 
                           f"Error processing CLA: {str(e)}")
            return False, f"Error processing CLA: {str(e)}"
    
    def approve_contributor(self, cla_id: str, admin_notes: str = "", 
                          approved_by: str = "system") -> bool:
        """Approve a contributor's CLA."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get contributor
            cursor.execute('SELECT * FROM contributors WHERE cla_id = ?', (cla_id,))
            contributor_data = cursor.fetchone()
            
            if not contributor_data:
                return False
            
            # Update status to approved
            cursor.execute('''
                UPDATE contributors 
                SET status = 'approved', 
                    approval_date = CURRENT_TIMESTAMP, 
                    notes = ?, 
                    updated_at = CURRENT_TIMESTAMP 
                WHERE cla_id = ?
            ''', (admin_notes, cla_id))
            
            conn.commit()
            conn.close()
            
            # Send approval email
            contributor = Contributor(*contributor_data[1:])  # Skip ID
            self._send_approval_email(contributor)
            
            # Update GitHub repository secrets
            self._update_github_secrets()
            
            # Log action
            self._log_action("contributor_approved", contributor_data[0], 
                           f"CLA {cla_id} approved by {approved_by}")
            
            return True
            
        except Exception as e:
            self._log_action("approval_error", None, 
                           f"Error approving CLA {cla_id}: {str(e)}")
            return False
    
    def reject_contributor(self, cla_id: str, reason: str, 
                          rejected_by: str = "system") -> bool:
        """Reject a contributor's CLA."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get contributor
            cursor.execute('SELECT * FROM contributors WHERE cla_id = ?', (cla_id,))
            contributor_data = cursor.fetchone()
            
            if not contributor_data:
                return False
            
            # Update status to rejected
            cursor.execute('''
                UPDATE contributors 
                SET status = 'rejected', 
                    notes = ?, 
                    updated_at = CURRENT_TIMESTAMP 
                WHERE cla_id = ?
            ''', (reason, cla_id))
            
            conn.commit()
            conn.close()
            
            # Send rejection email
            contributor = Contributor(*contributor_data[1:])
            self._send_rejection_email(contributor, reason)
            
            # Log action
            self._log_action("contributor_rejected", contributor_data[0], 
                           f"CLA {cla_id} rejected by {rejected_by}: {reason}")
            
            return True
            
        except Exception as e:
            self._log_action("rejection_error", None, 
                           f"Error rejecting CLA {cla_id}: {str(e)}")
            return False
    
    def get_contributor_status(self, identifier: str) -> Optional[Dict]:
        """Get contributor status by GitHub username or CLA ID."""
        contributor = None
        
        if identifier.startswith("CLA-"):
            contributor = self._get_contributor_by_cla_id(identifier)
        else:
            contributor = self._get_contributor_by_username(identifier)
        
        if contributor:
            return {
                "github_username": contributor.github_username,
                "legal_name": contributor.legal_name,
                "email": contributor.email,
                "cla_id": contributor.cla_id,
                "status": contributor.status,
                "cla_signed_date": contributor.cla_signed_date.isoformat() if contributor.cla_signed_date else None,
                "approval_date": contributor.approval_date.isoformat() if contributor.approval_date else None,
                "employer": contributor.employer,
                "witness_name": contributor.witness_name
            }
        
        return None
    
    def list_pending_contributors(self) -> List[Dict]:
        """List all contributors with pending CLA status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT github_username, legal_name, email, cla_id, cla_signed_date, employer
            FROM contributors 
            WHERE status = 'pending'
            ORDER BY cla_signed_date ASC
        ''')
        
        pending = []
        for row in cursor.fetchall():
            pending.append({
                "github_username": row[0],
                "legal_name": row[1],
                "email": row[2],
                "cla_id": row[3],
                "cla_signed_date": row[4],
                "employer": row[5],
                "days_pending": (datetime.now() - datetime.fromisoformat(row[4])).days if row[4] else 0
            })
        
        conn.close()
        return pending
    
    def check_cla_expiry(self) -> List[Dict]:
        """Check for contributors with expiring CLAs."""
        if self.config.get("cla_expiry_days", 0) <= 0:
            return []
        
        expiry_date = datetime.now() - timedelta(days=self.config["cla_expiry_days"])
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT github_username, legal_name, email, cla_id, approval_date
            FROM contributors 
            WHERE status = 'approved' AND approval_date < ?
            ORDER BY approval_date ASC
        ''', (expiry_date,))
        
        expiring = []
        for row in cursor.fetchall():
            expiring.append({
                "github_username": row[0],
                "legal_name": row[1],
                "email": row[2],
                "cla_id": row[3],
                "approval_date": row[4],
                "days_expired": (datetime.now() - datetime.fromisoformat(row[4])).days - self.config["cla_expiry_days"]
            })
        
        conn.close()
        return expiring
    
    def generate_contributor_report(self) -> Dict:
        """Generate comprehensive contributor statistics report."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Status breakdown
        cursor.execute('SELECT status, COUNT(*) FROM contributors GROUP BY status')
        status_breakdown = dict(cursor.fetchall())
        
        # Monthly submissions
        cursor.execute('''
            SELECT strftime('%Y-%m', cla_signed_date) as month, COUNT(*)
            FROM contributors
            WHERE cla_signed_date IS NOT NULL
            GROUP BY month
            ORDER BY month DESC
            LIMIT 12
        ''')
        monthly_submissions = dict(cursor.fetchall())
        
        # Recent activity
        cursor.execute('''
            SELECT github_username, legal_name, status, cla_signed_date
            FROM contributors
            ORDER BY cla_signed_date DESC
            LIMIT 10
        ''')
        recent_activity = [
            {
                "username": row[0],
                "name": row[1],
                "status": row[2],
                "date": row[3]
            }
            for row in cursor.fetchall()
        ]
        
        # Employer breakdown
        cursor.execute('''
            SELECT employer, COUNT(*)
            FROM contributors
            WHERE employer IS NOT NULL AND employer != ''
            GROUP BY employer
            ORDER BY COUNT(*) DESC
        ''')
        employer_breakdown = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "generated_at": datetime.now().isoformat(),
            "total_contributors": sum(status_breakdown.values()),
            "status_breakdown": status_breakdown,
            "monthly_submissions": monthly_submissions,
            "recent_activity": recent_activity,
            "employer_breakdown": employer_breakdown,
            "pending_count": status_breakdown.get("pending", 0),
            "approved_count": status_breakdown.get("approved", 0),
            "rejected_count": status_breakdown.get("rejected", 0)
        }
    
    def _validate_contributor_data(self, data: Dict) -> Tuple[bool, str]:
        """Validate contributor submission data."""
        required_fields = ['github_username', 'legal_name', 'email']
        
        for field in required_fields:
            if field not in data or not data[field].strip():
                return False, f"Missing required field: {field}"
        
        # Email validation
        email = data['email'].strip().lower()
        if '@' not in email or '.' not in email.split('@')[1]:
            return False, "Invalid email format"
        
        # Check blocked domains
        domain = email.split('@')[1]
        if domain in self.config.get("blocked_domains", []):
            return False, f"Email domain {domain} is not allowed"
        
        # GitHub username validation
        username = data['github_username'].strip()
        if not username.replace('-', '').replace('_', '').isalnum():
            return False, "Invalid GitHub username format"
        
        # Legal name validation
        legal_name = data['legal_name'].strip()
        if len(legal_name) < 2 or not any(c.isalpha() for c in legal_name):
            return False, "Legal name must contain at least 2 characters and include letters"
        
        # Witness requirement
        if self.config.get("require_witness", True):
            if not data.get('witness_name', '').strip():
                return False, "Witness name is required"
        
        return True, "Valid"
    
    def _generate_cla_id(self, username: str) -> str:
        """Generate unique CLA ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        random_suffix = secrets.token_hex(4).upper()
        return f"CLA-{timestamp}-{username.upper()[:8]}-{random_suffix}"
    
    def _should_auto_approve(self, email: str) -> bool:
        """Check if contributor should be auto-approved."""
        domain = email.split('@')[1].lower()
        return domain in self.config.get("auto_approve_domains", [])
    
    def _get_contributor_by_username(self, username: str) -> Optional[Contributor]:
        """Get contributor by GitHub username."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM contributors WHERE github_username = ?', (username,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return Contributor(
                id=row[0],
                github_username=row[1],
                legal_name=row[2],
                email=row[3],
                cla_signed_date=datetime.fromisoformat(row[4]) if row[4] else None,
                cla_id=row[5],
                status=row[6],
                employer=row[7],
                witness_name=row[8],
                document_hash=row[9],
                approval_date=datetime.fromisoformat(row[10]) if row[10] else None,
                notes=row[11]
            )
        
        return None
    
    def _get_contributor_by_cla_id(self, cla_id: str) -> Optional[Contributor]:
        """Get contributor by CLA ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM contributors WHERE cla_id = ?', (cla_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return Contributor(
                id=row[0],
                github_username=row[1],
                legal_name=row[2],
                email=row[3],
                cla_signed_date=datetime.fromisoformat(row[4]) if row[4] else None,
                cla_id=row[5],
                status=row[6],
                employer=row[7],
                witness_name=row[8],
                document_hash=row[9],
                approval_date=datetime.fromisoformat(row[10]) if row[10] else None,
                notes=row[11]
            )
        
        return None
    
    def _store_contributor(self, contributor: Contributor) -> int:
        """Store contributor in database and return ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO contributors 
            (github_username, legal_name, email, cla_signed_date, cla_id, 
             status, employer, witness_name, document_hash, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            contributor.github_username,
            contributor.legal_name,
            contributor.email,
            contributor.cla_signed_date.isoformat() if contributor.cla_signed_date else None,
            contributor.cla_id,
            contributor.status,
            contributor.employer,
            contributor.witness_name,
            contributor.document_hash,
            contributor.notes
        ))
        
        contributor_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return contributor_id
    
    def _log_action(self, action: str, contributor_id: Optional[int], 
                   details: str, performed_by: str = "system"):
        """Log action to audit trail."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_log (action, contributor_id, details, performed_by)
            VALUES (?, ?, ?, ?)
        ''', (action, contributor_id, details, performed_by))
        
        conn.commit()
        conn.close()
    
    def _send_cla_confirmation_email(self, contributor: Contributor):
        """Send CLA confirmation email."""
        subject = f"🔒 CLA Received - {contributor.cla_id}"
        
        html_content = f"""
        <h2>🔒 CLA Submission Received - Fire Prevention System</h2>
        
        <p>Dear {contributor.legal_name},</p>
        
        <p>We have received your signed Contributor License Agreement for the Fire Prevention System repository.</p>
        
        <h3>📋 Submission Details:</h3>
        <ul>
            <li><strong>CLA ID:</strong> {contributor.cla_id}</li>
            <li><strong>GitHub Username:</strong> {contributor.github_username}</li>
            <li><strong>Submission Date:</strong> {contributor.cla_signed_date.strftime('%Y-%m-%d %H:%M:%S UTC')}</li>
            <li><strong>Status:</strong> Pending Review</li>
            {f'<li><strong>Employer:</strong> {contributor.employer}</li>' if contributor.employer else ''}
            {f'<li><strong>Witness:</strong> {contributor.witness_name}</li>' if contributor.witness_name else ''}
        </ul>
        
        <h3>⏳ Next Steps:</h3>
        <ol>
            <li>Legal review of your CLA (2-5 business days)</li>
            <li>Verification of witness signature</li>
            <li>Background check (if applicable)</li>
            <li>Final approval notification</li>
        </ol>
        
        <h3>⚖️ Legal Reminder:</h3>
        <p>Upon approval, all your contributions to the Fire Prevention System will become the exclusive property of {self.config['owner_name']}. You will retain no ownership rights and receive no compensation.</p>
        
        <h3>📞 Contact:</h3>
        <p>Questions? Contact {self.config['legal_email']}</p>
        
        <hr>
        <p><em>Fire Prevention System - Proprietary License Enforcement System</em></p>
        """
        
        self._send_email(contributor.email, subject, html_content, "cla_confirmation")
    
    def _send_approval_email(self, contributor: Contributor):
        """Send contributor approval email."""
        subject = f"✅ CLA Approved - Welcome to Fire Prevention System"
        
        html_content = f"""
        <h2>✅ CLA Approved - Welcome to Fire Prevention System</h2>
        
        <p>Dear {contributor.legal_name},</p>
        
        <p>Congratulations! Your Contributor License Agreement has been <strong>approved</strong>.</p>
        
        <h3>📋 Approval Details:</h3>
        <ul>
            <li><strong>CLA ID:</strong> {contributor.cla_id}</li>
            <li><strong>GitHub Username:</strong> {contributor.github_username}</li>
            <li><strong>Approval Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</li>
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
        <p>Please review <a href="https://github.com/{self.config['repository']}/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a> for technical standards and development workflow.</p>
        
        <h3>⚖️ Legal Obligations:</h3>
        <ul>
            <li>All contributions become property of {self.config['owner_name']}</li>
            <li>No compensation or attribution guaranteed</li>
            <li>Commercial use rights assigned to owner</li>
            <li>Agreement is irrevocable and permanent</li>
        </ul>
        
        <p>Welcome to the team! We look forward to your contributions.</p>
        
        <hr>
        <p><em>Fire Prevention System Development Team</em></p>
        """
        
        self._send_email(contributor.email, subject, html_content, "cla_approval")
    
    def _send_rejection_email(self, contributor: Contributor, reason: str):
        """Send contributor rejection email."""
        subject = f"❌ CLA Not Approved - {contributor.cla_id}"
        
        html_content = f"""
        <h2>❌ CLA Not Approved - Fire Prevention System</h2>
        
        <p>Dear {contributor.legal_name},</p>
        
        <p>After review, we are unable to approve your Contributor License Agreement at this time.</p>
        
        <h3>📋 Review Details:</h3>
        <ul>
            <li><strong>CLA ID:</strong> {contributor.cla_id}</li>
            <li><strong>GitHub Username:</strong> {contributor.github_username}</li>
            <li><strong>Review Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</li>
            <li><strong>Status:</strong> ❌ NOT APPROVED</li>
        </ul>
        
        <h3>📝 Reason:</h3>
        <p>{reason}</p>
        
        <h3>🔄 Next Steps:</h3>
        <p>If you believe this decision was made in error or if you can address the concerns raised, please contact our legal team at {self.config['legal_email']}.</p>
        
        <h3>📞 Contact:</h3>
        <p>Legal Questions: {self.config['legal_email']}</p>
        
        <hr>
        <p><em>Fire Prevention System - Legal Department</em></p>
        """
        
        self._send_email(contributor.email, subject, html_content, "cla_rejection")
    
    def _send_email(self, to_email: str, subject: str, html_content: str, email_type: str):
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
            
            # Log email success
            self._log_email(to_email, subject, email_type, True)
            
        except Exception as e:
            # Log email failure
            self._log_email(to_email, subject, email_type, False)
            print(f"❌ Failed to send email to {to_email}: {e}")
    
    def _log_email(self, recipient: str, subject: str, email_type: str, success: bool):
        """Log email sending attempt."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO email_log (recipient_email, subject, email_type, success)
            VALUES (?, ?, ?, ?)
        ''', (recipient, subject, email_type, success))
        
        conn.commit()
        conn.close()
    
    def _update_github_secrets(self):
        """Update GitHub repository secrets with approved contributors."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT github_username FROM contributors WHERE status = "approved"')
            approved_usernames = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            # This would update GitHub secrets via API
            print(f"📊 Would update GitHub secrets with {len(approved_usernames)} approved contributors")
            
        except Exception as e:
            print(f"❌ Failed to update GitHub secrets: {e}")


def main():
    """Main entry point for CLA processor."""
    parser = argparse.ArgumentParser(description="CLA Processing and Management System")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--add-contributor", help="Add contributor from JSON file")
    parser.add_argument("--approve", help="Approve CLA by ID")
    parser.add_argument("--reject", help="Reject CLA by ID with reason", nargs=2, metavar=('CLA_ID', 'REASON'))
    parser.add_argument("--status", help="Check contributor status by username or CLA ID")
    parser.add_argument("--list-pending", action="store_true", help="List pending contributors")
    parser.add_argument("--check-expiry", action="store_true", help="Check for expiring CLAs")
    parser.add_argument("--report", action="store_true", help="Generate contributor report")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = CLAProcessor(args.config)
    
    if args.add_contributor:
        with open(args.add_contributor) as f:
            contributor_data = json.load(f)
        success, result = processor.process_new_cla(contributor_data)
        print(f"✅ CLA processed: {result}" if success else f"❌ Error: {result}")
    
    elif args.approve:
        success = processor.approve_contributor(args.approve)
        print(f"✅ CLA approved" if success else f"❌ Failed to approve CLA")
    
    elif args.reject:
        cla_id, reason = args.reject
        success = processor.reject_contributor(cla_id, reason)
        print(f"✅ CLA rejected" if success else f"❌ Failed to reject CLA")
    
    elif args.status:
        status = processor.get_contributor_status(args.status)
        if status:
            print(json.dumps(status, indent=2))
        else:
            print(f"❌ Contributor not found: {args.status}")
    
    elif args.list_pending:
        pending = processor.list_pending_contributors()
        print(f"📋 Pending Contributors ({len(pending)}):")
        for contributor in pending:
            print(f"  • {contributor['github_username']} ({contributor['cla_id']}) - {contributor['days_pending']} days")
    
    elif args.check_expiry:
        expiring = processor.check_cla_expiry()
        print(f"⏰ Expiring CLAs ({len(expiring)}):")
        for contributor in expiring:
            print(f"  • {contributor['github_username']} - expired {contributor['days_expired']} days ago")
    
    elif args.report:
        report = processor.generate_contributor_report()
        print(json.dumps(report, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
