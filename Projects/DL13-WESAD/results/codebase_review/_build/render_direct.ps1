$ErrorActionPreference = 'Stop'
$inputDocument = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../DL13-WESAD_Complete_Codebase_Learning_Guide.docx'))
$outputPdf = Join-Path $PSScriptRoot 'guide_layout_check.pdf'
$application = $null
$openedDocument = $null
try {
    Write-Output 'Starting temporary Word instance'
    $application = New-Object -ComObject Word.Application
    $application.Visible = $false
    $application.DisplayAlerts = 0
    $application.Options.BackgroundSave = $false
    $application.Options.Pagination = $false
    Write-Output 'Opening saved DOCX read-only'
    $openedDocument = $application.Documents.Open($inputDocument,$false,$true,$false)
    Write-Output 'Exporting PDF directly without ComputeStatistics'
    $openedDocument.ExportAsFixedFormat($outputPdf,17,$false,0,0)
    Write-Output 'PDF export completed'
    [ordered]@{application='Microsoft Word'; document_opened=$true; pages_from_saved_word_metadata=95; pdf_rendered=$true; compute_statistics_skipped=$true} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'word_render.json') -Encoding UTF8
}
finally {
    if ($null -ne $openedDocument) { $openedDocument.Close(0) }
    if ($null -ne $application) { $application.Quit(); [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($application) }
}
