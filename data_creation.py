from openpyxl import Workbook
from raw_data import travel_agency_issues, tele_communication_issues, visa_issues


def export_travel_issues_to_excel(travel_agency_issues, output_file="tele_issues.xlsx"):
    """
    Converts travel agency issues JSON data into an Excel file.

    Columns:
    - Serial Number (id)
    - Problem
    - Resolution
    """

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Travel Agency Issues"

    # Header row
    sheet.append(["Serial Number", "Problem", "Resolution"])

    # Data rows
    for item in travel_agency_issues:
        sheet.append([item.get("id"), item.get("problem"), item.get("resolution")])

    workbook.save(output_file)
    print(f"Excel file successfully created: {output_file}")


export_travel_issues_to_excel(tele_communication_issues)
