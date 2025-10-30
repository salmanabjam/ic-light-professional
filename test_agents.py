#!/usr/bin/env python3
"""
Comprehensive Test Suite for AI Dev Collective v9.0
Demonstrates all features and validates functionality
"""

import sys
from pathlib import Path

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


def test_agent_creation():
    """Test that all agents can be created"""
    print("Test 1: Agent Creation")
    
    agents = [
        AstroLeadDeveloper(),
        LyraResearchAssistant(),
        NexusCodeQualityAssistant(),
        CryptoXSecurityAnalyst(),
        NOVAUIUXDesigner(),
        EchoPerformanceAnalyst(),
        SageDocumentationSpecialist(),
        PulseDevOpsSpecialist()
    ]
    
    for agent in agents:
        assert agent.name is not None, f"Agent name is None"
        assert agent.role is not None, f"Agent role is None"
        assert len(agent.responsibilities) > 0, f"Agent has no responsibilities"
        print(f"  ✅ {agent.name} created successfully")
    
    print(f"  ✅ All {len(agents)} agents created successfully\n")
    return True


def test_team_creation():
    """Test team creation and member management"""
    print("Test 2: Team Creation and Management")
    
    team = AgentTeam(
        name="Test Team",
        description="Test team for validation"
    )
    
    assert team.name == "Test Team"
    assert len(team.members) == 0
    
    team.add_member(AstroLeadDeveloper())
    team.add_member(CryptoXSecurityAnalyst())
    
    assert len(team.members) == 2
    print(f"  ✅ Team created with {len(team.members)} members")
    print(f"  ✅ Members: {[m.name for m in team.members]}\n")
    return True


def test_analysis():
    """Test running analysis"""
    print("Test 3: Running Analysis")
    
    team = AgentTeam(
        name="Analysis Test Team",
        description="Team for testing analysis"
    )
    
    team.add_member(AstroLeadDeveloper())
    team.add_member(CryptoXSecurityAnalyst())
    
    # Get current directory
    project_path = Path(__file__).parent.absolute()
    
    print(f"  Running analysis on: {project_path}")
    results = team.run_analysis(str(project_path))
    
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    
    for agent_name, report in results.items():
        assert report is not None, f"Report is None for {agent_name}"
        assert hasattr(report, 'findings'), f"Report missing findings"
        assert hasattr(report, 'recommendations'), f"Report missing recommendations"
        print(f"  ✅ {agent_name}: {len(report.findings)} findings, {len(report.recommendations)} recommendations")
    
    print()
    return True


def test_summary_report():
    """Test summary report generation"""
    print("Test 4: Summary Report Generation")
    
    team = AgentTeam(
        name="Summary Test Team",
        description="Team for testing summary"
    )
    
    team.add_member(AstroLeadDeveloper())
    
    project_path = Path(__file__).parent.absolute()
    results = team.run_analysis(str(project_path))
    
    summary = team.generate_summary_report()
    
    assert 'team' in summary
    assert 'total_findings' in summary
    assert 'total_recommendations' in summary
    assert 'agents' in summary
    
    print(f"  ✅ Summary generated successfully")
    print(f"  ✅ Total findings: {summary['total_findings']}")
    print(f"  ✅ Total recommendations: {summary['total_recommendations']}")
    print()
    return True


def test_report_export():
    """Test report export functionality"""
    print("Test 5: Report Export")
    
    team = AgentTeam(
        name="Export Test Team",
        description="Team for testing export"
    )
    
    team.add_member(AstroLeadDeveloper())
    
    project_path = Path(__file__).parent.absolute()
    results = team.run_analysis(str(project_path))
    
    # Test markdown export
    output_file = "/tmp/test_report.md"
    team.export_reports(output_file, format="markdown")
    
    assert Path(output_file).exists(), "Report file not created"
    
    with open(output_file, 'r') as f:
        content = f.read()
        assert len(content) > 0, "Report is empty"
        assert "Astro" in content, "Agent name not in report"
    
    print(f"  ✅ Markdown report exported to {output_file}")
    print(f"  ✅ Report size: {len(content)} bytes")
    print()
    return True


def test_all_agents():
    """Test all agents on the project"""
    print("Test 6: Full Team Analysis")
    
    team = AgentTeam(
        name="AI Dev Collective v9.0",
        description="Full team for comprehensive analysis"
    )
    
    # Add all agents
    team.add_member(AstroLeadDeveloper())
    team.add_member(LyraResearchAssistant())
    team.add_member(NexusCodeQualityAssistant())
    team.add_member(CryptoXSecurityAnalyst())
    team.add_member(NOVAUIUXDesigner())
    team.add_member(EchoPerformanceAnalyst())
    team.add_member(SageDocumentationSpecialist())
    team.add_member(PulseDevOpsSpecialist())
    
    project_path = Path(__file__).parent.absolute()
    results = team.run_analysis(str(project_path))
    
    assert len(results) == 8, f"Expected 8 results, got {len(results)}"
    
    summary = team.generate_summary_report()
    
    print(f"  ✅ All 8 agents completed analysis")
    print(f"  ✅ Total findings: {summary['total_findings']}")
    print(f"  ✅ Total recommendations: {summary['total_recommendations']}")
    print(f"  ✅ Critical issues: {summary['critical_issues_count']}")
    print()
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("AI Dev Collective v9.0 - Comprehensive Test Suite")
    print("=" * 80)
    print()
    
    tests = [
        test_agent_creation,
        test_team_creation,
        test_analysis,
        test_summary_report,
        test_report_export,
        test_all_agents
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ❌ {test.__name__} failed\n")
        except Exception as e:
            failed += 1
            print(f"  ❌ {test.__name__} failed with error: {e}\n")
    
    print("=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("✨ All tests passed!")
        return 0
    else:
        print(f"⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
