$ErrorActionPreference = 'Stop'
$buildFolder = $PSScriptRoot
$documentPath = [System.IO.Path]::GetFullPath((Join-Path $buildFolder '../DL13-WESAD_Complete_Codebase_Learning_Guide.docx'))
$pdfPath = Join-Path $buildFolder 'guide_layout_check.pdf'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($documentPath, $false, $false, $false)
    $null = $document.Fields.Update()
    foreach ($contents in $document.TablesOfContents) { $null = $contents.Update() }
    $document.Repaginate()
    foreach ($contents in $document.TablesOfContents) { $null = $contents.UpdatePageNumbers() }
    $document.Save()
    $pages = $document.ComputeStatistics(2)
    $document.ExportAsFixedFormat($pdfPath, 17)
    $report = [ordered]@{
        application = 'Microsoft Word'
        document_opened = $true
        tables_of_contents = $document.TablesOfContents.Count
        tables = $document.Tables.Count
        inline_shapes = $document.InlineShapes.Count
        pages = $pages
        word_version = $word.Version
        pdf_rendered = (Test-Path -LiteralPath $pdfPath)
    }
    $report | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $buildFolder 'word_render.json') -Encoding UTF8
    $report | ConvertTo-Json
}
finally {
    if ($null -ne $document) { $document.Close(0) }
    if ($null -ne $word) { $word.Quit(); [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) }
}
