# green_parser.pl - Perl Regex for Greentext Ramps
#!/usr/bin/perl
use strict;
use warnings;
my $input = $ARGV[0] || '';
my @lines = split /\n/, $input;
my @ramp_code;
foreach my $line (@lines) {
    if ($line =~ /^>/) {
        my $verb = $line;
        $verb =~ s/^>//;
        $verb =~ s/^\s+|\s+$//g;
        $verb = lc($verb);
        push @ramp_code, "# $verb ramp";
    }
}
print join("\n", @ramp_code) . "\n";
