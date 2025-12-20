#!/usr/bin/env python3
"""
🔍 License Violation Detection System
Fire Prevention System Repository

This script automatically scans for potential license violations
across GitHub, commercial websites, and other platforms.

Author: [Your Full Legal Name]
License: Proprietary - All Rights Reserved
Version: 2.0
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict

import requests
from github import Github


@dataclass
class ViolationAlert:
    """License violation detection result."""
    violation_type: str
    detected_url: str
    confidence_score: float
    description: str
    evidence: Dict
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    
    def to_dict(self):
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


class LicenseViolationDetector:
    """Comprehensive license violation detection system."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize violation detector."""
        self.config = config or self._load_default_config()
        self.github = Github(self.config.get("github_token", os.getenv("GITHUB_TOKEN")))
        self.monitoring_keywords = self.config.get("monitoring_keywords", [])
        self.violations = []
        
    def _load_default_config(self) -> Dict:
        """Load default configuration."""
        return {
            "repository": "your-username/FIREPREVENTION",
            "owner_name": "[Your Full Legal Name]",
            "monitoring_keywords": [
                "fire detection yolo",
                "wildfire ai monitoring", 
                "drone fire prevention",
                "computer vision fire",
                "timesformer smoke detection",
                "resnet50 vari vegetation",
                "csrnet wildlife monitoring",
                "hybrid physics neural fire"
            ],
            "similarity_threshold": 0.7,
            "check_interval_hours": 6,
            "max_repos_per_search": 20
        }
    
    def run_comprehensive_scan(self) -> List[ViolationAlert]:
        """Run comprehensive violation detection scan."""
        print("🔍 Starting comprehensive license violation scan...")
        
        all_violations = []
        
        # 1. GitHub repository similarity scan
        print("📊 Scanning GitHub repositories...")
        github_violations = self._scan_github_repositories()
        all_violations.extend(github_violations)
        
        # 2. Commercial website scan
        print("🏢 Scanning commercial websites...")
        commercial_violations = self._scan_commercial_websites()
        all_violations.extend(commercial_violations)
        
        # 3. Academic paper scan
        print("🎓 Scanning academic publications...")
        academic_violations = self._scan_academic_publications()
        all_violations.extend(academic_violations)
        
        # 4. Social media and forums scan
        print("💬 Scanning social media and forums...")
        social_violations = self._scan_social_media()
        all_violations.extend(social_violations)
        
        # 5. Code repository platforms scan
        print("⚙️ Scanning other code platforms...")
        platform_violations = self._scan_code_platforms()
        all_violations.extend(platform_violations)
        
        self.violations = all_violations
        
        print(f"✅ Scan complete. Found {len(all_violations)} potential violations.")
        
        # Generate violation report
        if all_violations:
            self._generate_violation_report(all_violations)
            
        return all_violations
    
    def _scan_github_repositories(self) -> List[ViolationAlert]:
        """Scan GitHub for repositories with similar code or concepts."""
        violations = []
        
        for keyword in self.monitoring_keywords:
            try:
                print(f"  🔎 Searching for: '{keyword}'")
                repos = self.github.search_repositories(
                    query=keyword, 
                    sort="updated",
                    order="desc"
                )
                
                for i, repo in enumerate(repos):
                    if i >= self.config["max_repos_per_search"]:
                        break
                        
                    # Skip our own repository
                    if repo.full_name == self.config["repository"]:
                        continue
                    
                    # Analyze repository for violations
                    violation = self._analyze_repository_similarity(repo, keyword)
                    if violation:
                        violations.append(violation)
                        
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error searching for '{keyword}': {e}")
                
        return violations
    
    def _analyze_repository_similarity(self, repo, search_keyword: str) -> Optional[ViolationAlert]:
        """Analyze a repository for potential license violations."""
        try:
            similarity_score = 0.0
            evidence = {
                "repo_name": repo.full_name,
                "description": repo.description or "",
                "language": repo.language,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "created_at": repo.created_at.isoformat(),
                "updated_at": repo.updated_at.isoformat(),
                "search_keyword": search_keyword
            }
            
            # Check README content
            try:
                readme = repo.get_readme()
                readme_content = readme.decoded_content.decode('utf-8', errors='ignore').lower()
                
                # Look for specific proprietary terms
                proprietary_terms = [
                    "yolov8", "timesformer", "resnet50", "vari", "csrnet",
                    "fire detection", "smoke detection", "vegetation health",
                    "wildlife monitoring", "fire spread", "drone monitoring"
                ]
                
                matches = sum(1 for term in proprietary_terms if term in readme_content)
                similarity_score += matches * 0.15
                evidence["readme_matches"] = matches
                evidence["readme_terms_found"] = [term for term in proprietary_terms if term in readme_content]
                
                # Check for direct code copying indicators
                code_indicators = [
                    "fire prevention system", 
                    "hybrid physics neural",
                    "drone fire monitoring",
                    "autonomous wildfire"
                ]
                
                direct_matches = sum(1 for indicator in code_indicators if indicator in readme_content)
                similarity_score += direct_matches * 0.25
                evidence["direct_code_indicators"] = direct_matches
                
            except Exception as e:
                evidence["readme_error"] = str(e)
            
            # Check file structure
            try:
                contents = repo.get_contents("")
                file_names = [item.name.lower() for item in contents if item.type == "file"]
                
                # Look for similar file patterns from our project
                our_files = [
                    "train_fire.py", "train_smoke.py", "train_veg.py", 
                    "train_fauna.py", "simulate_spread.py", "monitor.py",
                    "benchmark.py", "validate_system.py"
                ]
                
                file_matches = sum(1 for our_file in our_files if our_file in file_names)
                similarity_score += file_matches * 0.2
                evidence["file_structure_matches"] = file_matches
                evidence["matching_files"] = [f for f in our_files if f in file_names]
                
            except Exception as e:
                evidence["file_structure_error"] = str(e)
            
            # Check repository description
            if repo.description:
                desc_lower = repo.description.lower()
                key_phrases = [
                    "fire detection", "wildfire", "drone monitoring", 
                    "computer vision", "ai fire", "forest fire"
                ]
                
                desc_matches = sum(1 for phrase in key_phrases if phrase in desc_lower)
                similarity_score += desc_matches * 0.1
                evidence["description_matches"] = desc_matches
            
            # Check license
            try:
                license_info = repo.get_license()
                if license_info:
                    evidence["license"] = license_info.license.name
                    
                    # Flag if using permissive license with our concepts
                    if license_info.license.name in ["MIT License", "Apache License 2.0"] and similarity_score > 0.3:
                        similarity_score += 0.2
                        evidence["license_conflict"] = True
                        
            except Exception as e:
                evidence["license_check_error"] = str(e)
            
            # Determine severity and create violation if threshold exceeded
            if similarity_score >= self.config["similarity_threshold"]:
                severity = self._calculate_severity(similarity_score, evidence)
                
                return ViolationAlert(
                    violation_type="code_similarity",
                    detected_url=repo.html_url,
                    confidence_score=min(similarity_score, 1.0),
                    description=f"Repository '{repo.full_name}' shows significant similarity to proprietary Fire Prevention System code",
                    evidence=evidence,
                    timestamp=datetime.now(),
                    severity=severity
                )
                
        except Exception as e:
            print(f"❌ Error analyzing repository {repo.full_name}: {e}")
            
        return None
    
    def _scan_commercial_websites(self) -> List[ViolationAlert]:
        """Scan commercial websites for unauthorized use."""
        violations = []
        
        # Commercial search queries
        commercial_queries = [
            "fire detection service company",
            "wildfire monitoring solution business",
            "ai fire prevention consulting",
            "drone fire detection product",
            "computer vision fire service"
        ]
        
        # This would integrate with web search APIs (Google, Bing, etc.)
        # For demonstration, we'll create placeholder logic
        
        print("  🏢 Commercial website scanning would require web search API integration")
        print("  💡 Consider integrating with Google Custom Search API, Bing Search API")
        
        return violations
    
    def _scan_academic_publications(self) -> List[ViolationAlert]:
        """Scan academic publications for proper attribution."""
        violations = []
        
        # Academic search terms
        academic_terms = [
            "fire detection yolo", "wildfire ai", "drone fire monitoring",
            "computer vision fire", "smoke detection transformer"
        ]
        
        # This would integrate with academic APIs (Google Scholar, arXiv, IEEE Xplore)
        print("  🎓 Academic publication scanning would require scholarly API integration")
        print("  💡 Consider integrating with arXiv API, Google Scholar, Semantic Scholar")
        
        return violations
    
    def _scan_social_media(self) -> List[ViolationAlert]:
        """Scan social media and forums for violations."""
        violations = []
        
        # Platforms to monitor: Reddit, Twitter, LinkedIn, Stack Overflow
        print("  💬 Social media scanning would require platform API integration")
        print("  💡 Consider integrating with Reddit API, Twitter API, LinkedIn API")
        
        return violations
    
    def _scan_code_platforms(self) -> List[ViolationAlert]:
        """Scan other code repository platforms."""
        violations = []
        
        # Other platforms: GitLab, Bitbucket, SourceForge, CodePen
        print("  ⚙️ Other platform scanning would require individual API integration")
        print("  💡 Consider integrating with GitLab API, Bitbucket API")
        
        return violations
    
    def _calculate_severity(self, similarity_score: float, evidence: Dict) -> str:
        """Calculate violation severity based on score and evidence."""
        if similarity_score >= 0.9:
            return "critical"
        elif similarity_score >= 0.8:
            return "high"
        elif similarity_score >= 0.7:
            return "medium"
        else:
            return "low"
    
    def _generate_violation_report(self, violations: List[ViolationAlert]):
        """Generate comprehensive violation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"violation_report_{timestamp}.json")
        
        report_data = {
            "scan_timestamp": datetime.now().isoformat(),
            "total_violations": len(violations),
            "severity_breakdown": {
                "critical": len([v for v in violations if v.severity == "critical"]),
                "high": len([v for v in violations if v.severity == "high"]),
                "medium": len([v for v in violations if v.severity == "medium"]),
                "low": len([v for v in violations if v.severity == "low"])
            },
            "violations": [v.to_dict() for v in violations],
            "recommendations": self._generate_recommendations(violations)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📊 Violation report saved to: {report_path}")
        
        # Generate human-readable summary
        self._generate_summary_report(violations, report_data)
    
    def _generate_recommendations(self, violations: List[ViolationAlert]) -> List[str]:
        """Generate recommendations based on detected violations."""
        recommendations = []
        
        critical_violations = [v for v in violations if v.severity == "critical"]
        high_violations = [v for v in violations if v.severity == "high"]
        
        if critical_violations:
            recommendations.append(
                "🚨 IMMEDIATE ACTION REQUIRED: Critical violations detected. "
                "Contact legal counsel immediately and prepare cease-and-desist notices."
            )
        
        if high_violations:
            recommendations.append(
                "⚠️ HIGH PRIORITY: High-confidence violations detected. "
                "Investigate immediately and document evidence for legal proceedings."
            )
        
        if len(violations) > 10:
            recommendations.append(
                "📈 PATTERN ALERT: Multiple violations detected. "
                "Consider increasing monitoring frequency and implementing automated takedown procedures."
            )
        
        # Repository-specific recommendations
        github_violations = [v for v in violations if v.violation_type == "code_similarity"]
        if github_violations:
            recommendations.append(
                "📋 GITHUB ACTION: Consider filing DMCA takedown requests for infringing repositories."
            )
        
        return recommendations
    
    def _generate_summary_report(self, violations: List[ViolationAlert], report_data: Dict):
        """Generate human-readable summary report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        summary = f"""
🔒 LICENSE VIOLATION DETECTION REPORT
Fire Prevention System Repository
Generated: {timestamp}

📊 SUMMARY STATISTICS:
• Total Violations: {len(violations)}
• Critical: {report_data['severity_breakdown']['critical']}
• High: {report_data['severity_breakdown']['high']}
• Medium: {report_data['severity_breakdown']['medium']}
• Low: {report_data['severity_breakdown']['low']}

🚨 TOP VIOLATIONS:
"""
        
        # Show top 5 highest confidence violations
        top_violations = sorted(violations, key=lambda v: v.confidence_score, reverse=True)[:5]
        
        for i, violation in enumerate(top_violations, 1):
            summary += f"""
{i}. {violation.violation_type.upper()} - {violation.severity.upper()}
   URL: {violation.detected_url}
   Confidence: {violation.confidence_score:.2f}
   Description: {violation.description}
"""
        
        summary += f"""

💡 RECOMMENDATIONS:
"""
        for rec in report_data['recommendations']:
            summary += f"• {rec}\n"
        
        summary += f"""

📞 NEXT STEPS:
1. Review detailed JSON report for complete evidence
2. Contact legal counsel for high/critical violations
3. Document all evidence for potential legal proceedings
4. Consider sending cease-and-desist notices
5. Monitor for compliance with takedown requests

---
Automated License Enforcement System
Fire Prevention System Repository
"""
        
        summary_path = Path(f"violation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"📋 Summary report saved to: {summary_path}")
        print("\n" + "="*60)
        print(summary)
        print("="*60)


def main():
    """Main entry point for violation detection."""
    parser = argparse.ArgumentParser(description="License Violation Detection System")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--automated-scan", action="store_true", help="Run automated scan")
    parser.add_argument("--github-only", action="store_true", help="Scan GitHub repositories only")
    parser.add_argument("--keywords", nargs="+", help="Custom keywords to search for")
    parser.add_argument("--output-dir", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
    
    # Override keywords if provided
    if args.keywords:
        config = config or {}
        config["monitoring_keywords"] = args.keywords
    
    # Initialize detector
    detector = LicenseViolationDetector(config)
    
    # Run scan
    if args.github_only:
        print("🔍 Running GitHub-only scan...")
        violations = detector._scan_github_repositories()
    else:
        print("🔍 Running comprehensive violation scan...")
        violations = detector.run_comprehensive_scan()
    
    # Output results
    if violations:
        print(f"\n⚠️ Found {len(violations)} potential violations!")
        
        # Show summary
        for violation in violations[:3]:  # Show top 3
            print(f"• {violation.violation_type}: {violation.detected_url}")
            print(f"  Confidence: {violation.confidence_score:.2f} | Severity: {violation.severity}")
    else:
        print("\n✅ No violations detected in this scan.")


if __name__ == "__main__":
    main()
