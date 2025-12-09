#!/usr/bin/env python3
"""
File Extension Corrector
Changes file extensions based on actual file content (magic bytes)
"""

import os
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

# Common file signatures (magic bytes) and their corresponding extensions
FILE_SIGNATURES = {
    'jpg': [(0x00, b'\xFF\xD8\xFF')],
    'png': [(0x00, b'\x89PNG\r\n\x1a\n')],
    'gif': [(0x00, b'GIF8')],
    'pdf': [(0x00, b'%PDF')],
    'zip': [(0x00, b'PK\x03\x04')],
    'rar': [(0x00, b'Rar!\x1a\x07')],
    'gz': [(0x00, b'\x1f\x8b\x08')],
    'bmp': [(0x00, b'\x42\x4d')],
    'tif': [(0x00, b'\x49\x49\x2a\x00')],  # TIFF little-endian
    'tif': [(0x00, b'\x4d\x4d\x00\x2a')],  # TIFF big-endian
    'webp': [(0x00, b'\x52\x49\x46\x46'),(0x06, b'\x00\x00\x57\x45\x42')],
    'zip': [(0x00, b'\x50\x4b\x03\x04')],
    'zip': [(0x00, b'\x50\x4b\x05\x06')],  # Empty ZIP
    'zip': [(0x00, b'\x50\x4b\x07\x08')],  # Spanned ZIP
    'ps': [(0x00, b'\x25\x21')],  # PostScript
    'rtf': [(0x00, b'\x7b\x5c\x72\x74\x66')],
    'ico': [(0x00, b'\x00\x00\x01\x00')],
    'cur': [(0x00, b'\x00\x00\x02\x00')],
    'msi': [(0x00, b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1')],  # MSI installer
    'exe': [(0x00, b'\x4d\x5a')],  # DOS MZ executable
    'eps': [(0x00, b'\x25\x50\x53\x2d\x41\x64\x6f\x62\x65')],  # EPS
    'flv': [(0x00, b'\x46\x4c\x56\x01')],  # FLV
    'avi': [(0x00, b'\x52\x49\x46\x46')],  # AVI
    'mp4': [(0x00, b'\x00\x00\x00\x20\x66\x74\x79\x70\x69\x73\x6F\x6D\x00\x00\x02\x00\x69\x73\x6F\x6D\x69\x73\x6F\x32\x61\x76\x63\x31\x6D\x70\x34\x31')],  # MP4
    'docx': [(0x00, b'\x50\x4b\x03\x04\x14\x00\x06\x00')],  # DOCX
    'xlsx': [(0x00, b'\x50\x4b\x03\x04\x14\x00\x08\x08')],  # XLSX
}

class FileExtensionCorrector:
    def __init__(self):
        self.console = Console()
        self.stats = {
            'processed': 0,
            'changed': 0,
            'errors': 0,
            'unknown': 0,
            'correct': 0,
            'file_types': defaultdict(int),
            'changes_by_type': defaultdict(int),
            'errors_list': []
        }

    def get_file_signature(self, file_path: Path, max_bytes: int = 40) -> Optional[bytes]:
        """Read the first few bytes of a file to get its signature"""
        try:
            with open(file_path, 'rb') as f:
                return f.read(max_bytes)
        except (IOError, OSError, PermissionError) as e:
            self.stats['errors'] += 1
            self.stats['errors_list'].append(f"{file_path}: {str(e)}")
            return None

    def is_text_file(self, file_path: Path) -> bool:
        """Check if a file is a text file by attempting to decode it"""
        try:
            # First check for common text file signatures
            signature = self.get_file_signature(file_path, 4)
            if signature:
                # Check for UTF-8 BOM
                if signature.startswith(b'\xef\xbb\xbf'):
                    return True
                # Check for UTF-16 BE BOM
                if signature.startswith(b'\xfe\xff'):
                    return True
                # Check for UTF-16 LE BOM
                if signature.startswith(b'\xff\xfe'):
                    return True
                # Check for UTF-32 BOMs
                if signature.startswith(b'\x00\x00\xfe\xff') or signature.startswith(b'\xff\xfe\x00\x00'):
                    return True
            
            # Try to decode as UTF-8 (most common)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read a sample and check if it contains mostly valid text characters
                sample = f.read(1024)
                if not sample:  # Empty file
                    return True
                
                # Count printable characters vs non-printable
                printable_count = sum(1 for char in sample if char.isprintable() or char in '\t\n\r')
                ratio = printable_count / len(sample)
                
                # If more than 80% of characters are printable, consider it text
                return ratio > 0.8
                
        except (UnicodeDecodeError, IOError, OSError, PermissionError, MemoryError):
            return False

    def detect_file_type(self, file_path: Path) -> Optional[str]:
        """Detect file type based on magic bytes"""
        signature = self.get_file_signature(file_path)
        if signature is None:
            return None
        
        # Check against known signatures
        for extension, magic_sequences in FILE_SIGNATURES.items():
            is_this_extension = True
            for offset, magic_bytes in magic_sequences:
                is_this_extension = is_this_extension and signature[offset:offset+len(magic_bytes)] == magic_bytes
            if is_this_extension:
                return extension
            # if signature.startswith(magic_bytes):
            #     return extension
        
        # Check for text files
        if self.is_text_file(file_path):
            return 'txt'
        
        return None

    def change_file_extension(self, file_path: Path, new_extension: str) -> bool:
        """Change the extension of a file"""
        try:
            new_path = file_path.with_suffix(f'.{new_extension}')
            
            # Check if target file already exists
            if new_path.exists():
                self.stats['errors'] += 1
                self.stats['errors_list'].append(f"{file_path}: Target file {new_path} already exists")
                return False
            
            file_path.rename(new_path)
            return True
        except (OSError, PermissionError) as e:
            self.stats['errors'] += 1
            self.stats['errors_list'].append(f"{file_path}: {str(e)}")
            return False

    def collect_files(self, directory_path: Path, recursive: bool = False) -> List[Path]:
        """Collect all files to be processed"""
        try:
            if recursive:
                files = [f for f in directory_path.rglob('*') if f.is_file()]
            else:
                files = [f for f in directory_path.glob('*') if f.is_file()]
            return files
        except (OSError, PermissionError) as e:
            self.stats['errors'] += 1
            self.stats['errors_list'].append(f"{directory_path}: {str(e)}")
            return []

    def process_files(self, files: List[Path], dry_run: bool = False) -> None:
        """Process all files with progress tracking"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
            transient=False,
        ) as progress:
            task = progress.add_task("Processing files...", total=len(files))
            
            for file_path in files:
                self.stats['processed'] += 1
                
                try:
                    current_extension = file_path.suffix.lower().lstrip('.')
                    detected_type = self.detect_file_type(file_path)
                    
                    if detected_type:
                        self.stats['file_types'][detected_type] += 1
                        
                        if current_extension != detected_type:
                            if not dry_run:
                                success = self.change_file_extension(file_path, detected_type)
                                if success:
                                    self.stats['changed'] += 1
                                    self.stats['changes_by_type'][detected_type] += 1
                            else:
                                self.stats['changed'] += 1
                                self.stats['changes_by_type'][detected_type] += 1
                        else:
                            self.stats['correct'] += 1
                    else:
                        self.stats['unknown'] += 1
                        self.stats['file_types']['unknown'] += 1
                
                except Exception as e:
                    self.stats['errors'] += 1
                    self.stats['errors_list'].append(f"{file_path}: Unexpected error - {str(e)}")
                
                progress.update(task, advance=1)

    def display_summary(self, dry_run: bool = False, time_taken: float = 0) -> None:
        """Display a comprehensive summary of the operation"""
        
        # Main summary table
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan", width=20)
        summary_table.add_column("Count", style="white", justify="right")
        summary_table.add_column("Percentage", style="green", justify="right")
        
        total_files = self.stats['processed']
        
        summary_table.add_row(
            "Total Files Processed", 
            str(total_files), 
            "100%"
        )
        summary_table.add_row(
            "Extensions Corrected", 
            str(self.stats['changed']), 
            f"{(self.stats['changed']/total_files*100):.1f}%" if total_files > 0 else "0%"
        )
        summary_table.add_row(
            "Already Correct", 
            str(self.stats['correct']), 
            f"{(self.stats['correct']/total_files*100):.1f}%" if total_files > 0 else "0%"
        )
        summary_table.add_row(
            "Unknown File Types", 
            str(self.stats['unknown']), 
            f"{(self.stats['unknown']/total_files*100):.1f}%" if total_files > 0 else "0%"
        )
        summary_table.add_row(
            "Errors", 
            str(self.stats['errors']), 
            f"{(self.stats['errors']/total_files*100):.1f}%" if total_files > 0 else "0%"
        )
        summary_table.add_row(
            "Time Taken", 
            f"{time_taken:.2f}s", 
            ""
        )
        
        # File type distribution table
        if self.stats['file_types']:
            type_table = Table(show_header=True, header_style="bold blue")
            type_table.add_column("File Type", style="cyan", width=12)
            type_table.add_column("Files Found", style="white", justify="right")
            type_table.add_column("Extensions Changed", style="yellow", justify="right")
            
            for file_type, count in sorted(self.stats['file_types'].items(), key=lambda x: x[1], reverse=True):
                changes = self.stats['changes_by_type'][file_type]
                type_table.add_row(
                    file_type.upper(),
                    str(count),
                    str(changes)
                )
        
        # Display everything
        self.console.print()
        
        # Title
        title = "📁 FILE EXTENSION CORRECTION SUMMARY"
        if dry_run:
            title += " - DRY RUN"
        self.console.print(Panel.fit(title, style="bold green" if not dry_run else "bold yellow"))
        
        # Summary
        self.console.print(Panel(summary_table, title="📊 Summary", border_style="green"))
        
        # File types
        if self.stats['file_types']:
            self.console.print(Panel(type_table, title="📄 File Types Distribution", border_style="blue"))
        
        # Warnings and errors
        if self.stats['errors_list']:
            error_panel = Panel(
                "\n".join([f"• {error}" for error in self.stats['errors_list'][:10]]),  # Show first 10 errors
                title=f"⚠️  Errors ({len(self.stats['errors_list'])} total)",
                border_style="red"
            )
            self.console.print(error_panel)
            with open('error_log.txt', 'w') as f:
                for error in self.stats['errors_list']:
                    f.write(f"{error}\n")
        
        # Final status
        self.console.print()
        if dry_run:
            status_text = Text("✅ DRY RUN COMPLETED - No files were modified", style="bold yellow")
        else:
            if self.stats['errors'] == 0:
                status_text = Text("✅ OPERATION COMPLETED SUCCESSFULLY", style="bold green")
            else:
                status_text = Text("⚠️  OPERATION COMPLETED WITH ERRORS", style="bold red")
        
        self.console.print(Panel.fit(status_text, border_style="yellow" if dry_run else "green"))

    def run(self, directory_path: str, dry_run: bool = False, recursive: bool = False) -> None:
        """Main execution method"""
        start_time = time.time()
        
        directory = Path(directory_path)
        
        if not directory.exists():
            self.console.print(f"[red]Error: Directory '{directory_path}' does not exist[/red]")
            return
        
        if not directory.is_dir():
            self.console.print(f"[red]Error: '{directory_path}' is not a directory[/red]")
            return
        
        # Collect files
        with self.console.status("[bold green]Collecting files...") as status:
            files = self.collect_files(directory, recursive)
        
        if not files:
            self.console.print(f"[yellow]No files found in '{directory_path}'[/yellow]")
            return
        
        self.console.print(f"[bold blue]Found {len(files)} files to process[/bold blue]")
        if dry_run:
            self.console.print("[bold yellow]Dry run mode - no files will be modified[/bold yellow]")
        
        # Process files
        self.process_files(files, dry_run)
        
        # Display summary
        time_taken = time.time() - start_time
        self.display_summary(dry_run, time_taken)

def main():
    parser = argparse.ArgumentParser(
        description='Change file extensions based on actual file content',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s /path/to/directory
  %(prog)s /path/to/directory --dry-run
  %(prog)s /path/to/directory --recursive
  %(prog)s /path/to/directory --dry-run --recursive
        '''
    )
    
    parser.add_argument('directory', help='Directory to process')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be changed without actually making changes')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Process subdirectories recursively')
    
    args = parser.parse_args()
    
    corrector = FileExtensionCorrector()
    corrector.run(args.directory, args.dry_run, args.recursive)

if __name__ == '__main__':
    main()