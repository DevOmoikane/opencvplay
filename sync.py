#!/usr/bin/env python3
"""
FTP-Local File Synchronizer
Deletes files that don't exist on the other side based on the specified side
"""

import os
import argparse
import ftplib
import sys
from pathlib import Path
from typing import Set, List, Tuple
from collections import defaultdict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

class FTPSynchronizer:
    def __init__(self):
        self.console = Console()
        self.stats = {
            'local_files': 0,
            'ftp_files': 0,
            'files_to_delete': 0,
            'deleted_files': 0,
            'errors': 0,
            'errors_list': []
        }

    def connect_ftp(self, host: str, username: str, password: str, ftp_path: str) -> ftplib.FTP:
        """Connect to FTP server and change to specified path"""
        try:
            self.console.print(f"[bold blue]Connecting to FTP server: {host}[/bold blue]")
            ftp = ftplib.FTP(host)
            ftp.login(username, password)
            
            # Change to specified path
            if ftp_path and ftp_path != '/':
                try:
                    ftp.cwd(ftp_path)
                    self.console.print(f"[bold green]Changed to FTP path: {ftp_path}[/bold green]")
                except ftplib.error_perm as e:
                    self.stats['errors'] += 1
                    self.stats['errors_list'].append(f"FTP path error: {str(e)}")
                    raise
            
            return ftp
        except Exception as e:
            self.stats['errors'] += 1
            self.stats['errors_list'].append(f"FTP connection error: {str(e)}")
            raise

    def get_local_files(self, local_path: str) -> Set[str]:
        """Get all files from local directory (recursively)"""
        local_dir = Path(local_path)
        if not local_dir.exists():
            raise FileNotFoundError(f"Local directory does not exist: {local_path}")
        
        if not local_dir.is_dir():
            raise NotADirectoryError(f"Local path is not a directory: {local_path}")
        
        files = set()
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                # Get relative path from local directory
                relative_path = file_path.relative_to(local_dir)
                files.add(str(relative_path))
        
        self.stats['local_files'] = len(files)
        return files

    def get_ftp_files(self, ftp: ftplib.FTP, base_path: str = '') -> Set[str]:
        """Recursively get all files from FTP server"""
        files = set()
        
        try:
            # Get list of items in current directory
            items = []
            ftp.retrlines(f'NLST {base_path}', items.append)
            
            for item in items:
                # Skip current and parent directory references
                if item in ['.', '..']:
                    continue
                
                full_path = os.path.join(base_path, item).replace('\\', '/')
                
                try:
                    # Try to change to directory - if it works, it's a directory
                    original_cwd = ftp.pwd()
                    ftp.cwd(item)
                    ftp.cwd(original_cwd)
                    # It's a directory, recurse into it
                    files.update(self.get_ftp_files(ftp, full_path))
                except ftplib.error_perm:
                    # It's a file, add to set
                    files.add(full_path)
                    
        except Exception as e:
            self.stats['errors'] += 1
            self.stats['errors_list'].append(f"FTP list error in {base_path}: {str(e)}")
        
        self.stats['ftp_files'] = len(files)
        return files

    def delete_local_files(self, files_to_delete: Set[str], local_path: str, dry_run: bool = False) -> None:
        """Delete local files that don't exist on FTP"""
        local_dir = Path(local_path)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
            transient=False,
        ) as progress:
            task = progress.add_task("Deleting local files...", total=len(files_to_delete))
            
            for file_rel_path in files_to_delete:
                file_path = local_dir / file_rel_path
                try:
                    if not dry_run:
                        file_path.unlink()  # Delete the file
                        self.stats['deleted_files'] += 1
                    progress.update(task, advance=1, description=f"Deleting: {file_rel_path}")
                except Exception as e:
                    self.stats['errors'] += 1
                    self.stats['errors_list'].append(f"Local delete error {file_rel_path}: {str(e)}")
                    progress.update(task, advance=1)

    def delete_ftp_files(self, ftp: ftplib.FTP, files_to_delete: Set[str], dry_run: bool = False) -> None:
        """Delete FTP files that don't exist locally"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
            transient=False,
        ) as progress:
            task = progress.add_task("Deleting FTP files...", total=len(files_to_delete))
            
            for file_path in files_to_delete:
                try:
                    if not dry_run:
                        ftp.delete(file_path)
                        self.stats['deleted_files'] += 1
                    progress.update(task, advance=1, description=f"Deleting: {file_path}")
                except Exception as e:
                    self.stats['errors'] += 1
                    self.stats['errors_list'].append(f"FTP delete error {file_path}: {str(e)}")
                    progress.update(task, advance=1)

    def display_comparison(self, local_files: Set[str], ftp_files: Set[str], side: str) -> None:
        """Display file comparison results"""
        only_local = local_files - ftp_files
        only_ftp = ftp_files - local_files
        common_files = local_files & ftp_files
        
        comparison_table = Table(show_header=True, header_style="bold cyan")
        comparison_table.add_column("File Type", style="white", width=20)
        comparison_table.add_column("Count", style="green", justify="right")
        
        comparison_table.add_row("Total Local Files", str(len(local_files)))
        comparison_table.add_row("Total FTP Files", str(len(ftp_files)))
        comparison_table.add_row("Common Files", str(len(common_files)))
        comparison_table.add_row("Only on Local", str(len(only_local)))
        comparison_table.add_row("Only on FTP", str(len(only_ftp)))
        
        self.console.print(Panel(comparison_table, title="📊 File Comparison", border_style="blue"))
        
        # Show what will be deleted
        if side == 'local':
            files_to_delete = only_local
            deletion_side = "LOCAL"
        else:  # side == 'ftp'
            files_to_delete = only_ftp
            deletion_side = "FTP"
        
        self.stats['files_to_delete'] = len(files_to_delete)
        
        if files_to_delete:
            files_list = "\n".join([f"• {f}" for f in sorted(list(files_to_delete))[:10]])  # Show first 10
            if len(files_to_delete) > 10:
                files_list += f"\n• ... and {len(files_to_delete) - 10} more files"
            
            self.console.print(Panel(
                files_list,
                title=f"🗑️  Files to delete from {deletion_side} ({len(files_to_delete)} files)",
                border_style="red"
            ))
        else:
            self.console.print(Panel(
                "No files to delete - both sides are already synchronized",
                title="✅ Synchronization Status",
                border_style="green"
            ))

    def display_summary(self, dry_run: bool = False, time_taken: float = 0) -> None:
        """Display operation summary"""
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan", width=25)
        summary_table.add_column("Count", style="white", justify="right")
        
        summary_table.add_row("Total Local Files", str(self.stats['local_files']))
        summary_table.add_row("Total FTP Files", str(self.stats['ftp_files']))
        summary_table.add_row("Files Scheduled for Deletion", str(self.stats['files_to_delete']))
        summary_table.add_row("Files Actually Deleted", str(self.stats['deleted_files']))
        summary_table.add_row("Errors Encountered", str(self.stats['errors']))
        summary_table.add_row("Time Taken", f"{time_taken:.2f}s")
        
        self.console.print()
        self.console.print(Panel(summary_table, title="📈 Operation Summary", border_style="green"))
        
        # Errors
        if self.stats['errors_list']:
            error_panel = Panel(
                "\n".join([f"• {error}" for error in self.stats['errors_list'][:10]]),
                title=f"⚠️  Errors ({len(self.stats['errors_list'])} total)",
                border_style="red"
            )
            self.console.print(error_panel)
        
        # Final status
        self.console.print()
        if dry_run:
            status_text = Text("✅ DRY RUN COMPLETED - No files were actually deleted", style="bold yellow")
        else:
            if self.stats['errors'] == 0:
                status_text = Text("✅ SYNCHRONIZATION COMPLETED SUCCESSFULLY", style="bold green")
            else:
                status_text = Text("⚠️  SYNCHRONIZATION COMPLETED WITH ERRORS", style="bold red")
        
        self.console.print(Panel.fit(status_text, border_style="yellow" if dry_run else "green"))

    def run(self, local_path: str, ftp_host: str, ftp_user: str, ftp_pass: str, 
            ftp_path: str, side: str, dry_run: bool = False) -> None:
        """Main synchronization method"""
        import time
        start_time = time.time()
        
        self.console.print(Panel.fit(
            f"FTP-Local File Synchronizer\n"
            f"Local: {local_path}\n"
            f"FTP: {ftp_user}@{ftp_host}:{ftp_path}\n"
            f"Side: {side.upper()}\n"
            f"Mode: {'DRY RUN' if dry_run else 'LIVE'}",
            style="bold blue"
        ))
        
        try:
            # Connect to FTP
            ftp = self.connect_ftp(ftp_host, ftp_user, ftp_pass, ftp_path)
            
            # Get file lists
            with self.console.status("[bold green]Scanning local files...[/bold green]") as status:
                local_files = self.get_local_files(local_path)
            
            with self.console.status("[bold green]Scanning FTP files...[/bold green]") as status:
                ftp_files = self.get_ftp_files(ftp)
            
            # Display comparison
            self.display_comparison(local_files, ftp_files, side)
            
            # Perform deletion based on side
            if side == 'local':
                files_to_delete = local_files - ftp_files
                if files_to_delete and not dry_run:
                    self.console.print("[bold red]Deleting local files not on FTP...[/bold red]")
                    self.delete_local_files(files_to_delete, local_path, dry_run)
                elif files_to_delete and dry_run:
                    self.console.print("[bold yellow]Dry run: Would delete local files not on FTP[/bold yellow]")
                    
            elif side == 'ftp':
                files_to_delete = ftp_files - local_files
                if files_to_delete and not dry_run:
                    self.console.print("[bold red]Deleting FTP files not on local...[/bold red]")
                    self.delete_ftp_files(ftp, files_to_delete, dry_run)
                elif files_to_delete and dry_run:
                    self.console.print("[bold yellow]Dry run: Would delete FTP files not on local[/bold yellow]")
            
            # Close FTP connection
            ftp.quit()
            
        except Exception as e:
            self.stats['errors'] += 1
            self.stats['errors_list'].append(f"Critical error: {str(e)}")
            self.console.print(f"[red]Critical error: {str(e)}[/red]")
        
        # Display final summary
        time_taken = time.time() - start_time
        self.display_summary(dry_run, time_taken)

def main():
    parser = argparse.ArgumentParser(
        description='Synchronize files between local directory and FTP server by deleting files not present on the other side',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Delete local files that don't exist on FTP (dry run)
  %(prog)s /local/path --user myuser --pass mypass --host ftp.example.com --path /remote/path --side local --dry-run
  
  # Delete FTP files that don't exist locally
  %(prog)s /local/path --user myuser --pass mypass --host ftp.example.com --path /remote/path --side ftp
  
  # Delete local files (live mode)
  %(prog)s /local/path --user myuser --pass mypass --host 192.168.1.100 --path /uploads --side local
        '''
    )
    
    parser.add_argument('local_path', help='Local directory path')
    parser.add_argument('--host', required=True, help='FTP server hostname or IP address')
    parser.add_argument('--user', required=True, help='FTP username')
    parser.add_argument('--pass', required=True, dest='password', help='FTP password')
    parser.add_argument('--path', default='', help='FTP server path (default: root directory)')
    parser.add_argument('--side', required=True, choices=['local', 'ftp'], 
                       help='Which side to delete files from (local: delete local files not on FTP, ftp: delete FTP files not on local)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be deleted without actually deleting anything')
    
    args = parser.parse_args()
    
    synchronizer = FTPSynchronizer()
    synchronizer.run(
        local_path=args.local_path,
        ftp_host=args.host,
        ftp_user=args.user,
        ftp_pass=args.password,
        ftp_path=args.path,
        side=args.side,
        dry_run=args.dry_run
    )

if __name__ == '__main__':
    main()