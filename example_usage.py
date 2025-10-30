"""
Example: Integrating AI Dev Collective with IC Light Application

This example demonstrates how to use the multi-agent analysis system
to analyze and improve the IC Light application.
"""

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


def main():
    """Run a comprehensive analysis of the IC Light project"""
    
    print("=" * 80)
    print("IC Light Professional - Software Analysis Example")
    print("=" * 80)
    print()
    
    # Create the AI Dev Collective team
    team = AgentTeam(
        name="AI Dev Collective v9.0 for Software Analysis & Enhancement",
        description="Multi-agent team dedicated to the deep analysis, research, and enhancement of IC Light software"
    )
    
    # Add all team members
    print("📋 Assembling the team...")
    team.add_member(AstroLeadDeveloper())
    team.add_member(LyraResearchAssistant())
    team.add_member(NexusCodeQualityAssistant())
    team.add_member(CryptoXSecurityAnalyst())
    team.add_member(NOVAUIUXDesigner())
    team.add_member(EchoPerformanceAnalyst())
    team.add_member(SageDocumentationSpecialist())
    team.add_member(PulseDevOpsSpecialist())
    
    print(f"✅ Team assembled: {len(team.members)} agents ready")
    print()
    
    # Run analysis on current project
    print("🔍 Running comprehensive analysis...")
    print()
    
    # Get current directory (the IC Light project)
    import os
    project_path = os.path.dirname(os.path.abspath(__file__))
    
    # Run the analysis
    results = team.run_analysis(project_path)
    
    print()
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print()
    
    # Display summary
    summary = team.generate_summary_report()
    
    print("📊 Executive Summary:")
    print(f"  Total Findings: {summary['total_findings']}")
    print(f"  Total Recommendations: {summary['total_recommendations']}")
    print(f"  Critical Issues: {summary['critical_issues_count']}")
    print()
    
    # Display critical issues if any
    if summary['critical_issues']:
        print("⚠️  Critical Issues Requiring Immediate Attention:")
        for i, issue in enumerate(summary['critical_issues'], 1):
            print(f"  {i}. {issue}")
        print()
    
    # Display highlights from each agent
    print("🔍 Agent Highlights:")
    print()
    
    for agent_name, report in results.items():
        print(f"  {agent_name} ({report.role}):")
        
        # Show priority
        priority_emoji = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        print(f"    Priority: {priority_emoji.get(report.priority, '⚪')} {report.priority.upper()}")
        
        # Show key findings (first 3)
        if report.findings:
            print(f"    Top Findings:")
            for finding in report.findings[:3]:
                print(f"      • {finding}")
        
        # Show key recommendations (first 2)
        if report.recommendations:
            print(f"    Key Recommendations:")
            for rec in report.recommendations[:2]:
                print(f"      → {rec}")
        
        print()
    
    # Export detailed reports
    print("💾 Exporting reports...")
    team.export_reports("analysis_report_detailed.md", format="markdown")
    print("   ✅ Markdown report saved: analysis_report_detailed.md")
    
    # Create action items
    print()
    print("📋 Suggested Action Items:")
    print()
    
    action_items = [
        ("HIGH", "Code Quality", "Add docstrings to all public functions and classes"),
        ("HIGH", "Security", "Review and secure any hardcoded credentials"),
        ("MEDIUM", "Performance", "Profile GPU usage and optimize memory allocation"),
        ("MEDIUM", "Documentation", "Create comprehensive API documentation"),
        ("MEDIUM", "DevOps", "Set up CI/CD pipeline with GitHub Actions"),
        ("LOW", "UI/UX", "Enhance Gradio interface with better error messages"),
    ]
    
    for priority, category, action in action_items:
        priority_color = {
            'HIGH': '🔴',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }
        print(f"  {priority_color[priority]} [{priority}] {category}: {action}")
    
    print()
    print("=" * 80)
    print("✨ Analysis complete! Use the detailed reports to guide improvements.")
    print("=" * 80)


if __name__ == "__main__":
    main()
