#!/usr/bin/env python3
"""
AI Dev Collective v9.0 CLI
Command-line interface for running software analysis
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from agents import (
    AgentTeam,
    AstroLeadDeveloper,
    LyraResearchAssistant,
    NexusCodeQualityAssistant,
    CryptoXSecurityAnalyst,
    NOVAUIUXDesigner,
    EchoPerformanceAnalyst,
    SageDocumentationSpecialist,
    PulseDevOpsSpecialist
)


def create_team():
    """Create and configure the AI Dev Collective team"""
    team = AgentTeam(
        name="AI Dev Collective v9.0 for Software Analysis & Enhancement",
        description="Multi-agent team dedicated to the deep analysis, research, and enhancement of any given software, ensuring improvements in code quality, security, performance, and user experience."
    )
    
    # Add all team members
    team.add_member(AstroLeadDeveloper())
    team.add_member(LyraResearchAssistant())
    team.add_member(NexusCodeQualityAssistant())
    team.add_member(CryptoXSecurityAnalyst())
    team.add_member(NOVAUIUXDesigner())
    team.add_member(EchoPerformanceAnalyst())
    team.add_member(SageDocumentationSpecialist())
    team.add_member(PulseDevOpsSpecialist())
    
    return team


def main():
    parser = argparse.ArgumentParser(
        description="AI Dev Collective v9.0 - Software Analysis & Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze current directory with all agents
  python analyze_software.py
  
  # Analyze specific directory
  python analyze_software.py --path /path/to/project
  
  # Run specific agents only
  python analyze_software.py --agents Astro CryptoX
  
  # Generate full report
  python analyze_software.py --output report.md --format markdown
        """
    )
    
    parser.add_argument(
        '--path',
        type=str,
        default='.',
        help='Path to the software project (default: current directory)'
    )
    
    parser.add_argument(
        '--agents',
        nargs='+',
        choices=['Astro', 'Lyra', 'Nexus', 'CryptoX', 'NOVA', 'Echo', 'Sage', 'Pulse'],
        help='Specific agents to run (default: all)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for the report'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'markdown'],
        default='markdown',
        help='Output format (default: markdown)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate project path
    project_path = Path(args.path).resolve()
    if not project_path.exists():
        print(f"❌ Error: Path '{project_path}' does not exist")
        sys.exit(1)
    
    # Create team
    print("🚀 Initializing AI Dev Collective v9.0...")
    print()
    team = create_team()
    print(f"✅ Team assembled with {len(team.members)} members")
    print()
    
    # Display team members
    if args.verbose:
        print("👥 Team Members:")
        for member in team.members:
            print(f"  • {member.name} - {member.role}")
        print()
    
    # Run analysis
    print(f"🔍 Analyzing project at: {project_path}")
    print("=" * 80)
    print()
    
    results = team.run_analysis(str(project_path), agents=args.agents)
    
    print()
    print("=" * 80)
    print(f"✅ Analysis complete! {len(results)} agent(s) completed their analysis")
    print()
    
    # Generate summary
    summary = team.generate_summary_report()
    
    print("📊 Summary:")
    print(f"  • Total findings: {summary['total_findings']}")
    print(f"  • Total recommendations: {summary['total_recommendations']}")
    print(f"  • Critical issues: {summary['critical_issues_count']}")
    print()
    
    # Display critical issues
    if summary['critical_issues']:
        print("⚠️ Critical Issues:")
        for issue in summary['critical_issues'][:5]:
            print(f"  • {issue}")
        print()
    
    # Display individual agent results
    if args.verbose:
        print("📋 Detailed Results:")
        print()
        for agent_name, report in results.items():
            print(f"🔸 {agent_name} ({report.role})")
            print(f"   Priority: {report.priority.upper()}")
            print(f"   Findings: {len(report.findings)}")
            print(f"   Recommendations: {len(report.recommendations)}")
            print()
    
    # Export report if requested
    if args.output:
        output_path = Path(args.output)
        team.export_reports(str(output_path), format=args.format)
        print(f"💾 Report exported to: {output_path}")
        print()
    
    # Display recommendations
    print("💡 Top Recommendations:")
    rec_count = 0
    for agent_name, report in results.items():
        for rec in report.recommendations[:2]:  # Top 2 from each agent
            rec_count += 1
            print(f"  {rec_count}. [{agent_name}] {rec}")
            if rec_count >= 10:
                break
        if rec_count >= 10:
            break
    print()
    
    print("✨ Analysis complete! Review the detailed reports for more information.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
