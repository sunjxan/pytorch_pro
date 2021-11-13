import re

def createNetClass(modulesString, num_classes=None, para_pth_path=None):
    ONE_SPACE = ' '
    IN_TAB_WIDTH = 2
    OUT_TAB_WIDTH = 4
    IN_TAB = ONE_SPACE * IN_TAB_WIDTH
    OUT_TAB = ONE_SPACE * OUT_TAB_WIDTH

    lines = modulesString.strip().split('\n')
    size = len(lines)
    header = '''class {:s}(nn.Module):\n{:s}def __init__(self):\n{:s}super().__init__()\n'''.format(lines[0][:-1], OUT_TAB, 2 * OUT_TAB)
    sections = []
    sectionBegin = None
    for i in range(1, size):
        line = lines[i]
        if line.startswith(IN_TAB + '('):
            if sectionBegin is not None:
                sections.append((sectionBegin, sectionBegin))
            sectionBegin = i
        elif line.startswith(IN_TAB + ')'):
            sections.append((sectionBegin, i))
            sectionBegin = None

    pattern = re.compile('^\s*\(([^)]+)\): (.+)$')
    body = ''
    funcs = []
    for first, last in sections:
        matches = pattern.match(lines[first])
        groups = matches.groups()
        body += '{:s}self.{:s} = nn.{:s}\n'.format(2 * OUT_TAB, *groups)
        funcs.append('self.{:s}'.format(groups[0]))
        if first != last:
            for i in range(first + 1, last):
                matches = pattern.match(lines[i])
                groups = matches.groups()
                if num_classes is not None and i == size - 3:
                    begin = groups[1].find('out_features=') + len('out_features=')
                    end = groups[1].find(',', begin)
                    body += '{:s}nn.{:s}{:d}{:s}{:s}\n'.format(3 * OUT_TAB, groups[1][:begin], num_classes, groups[1][end:], '' if i == last - 1 else ',')
                else:
                    body += '{:s}nn.{:s}{:s}\n'.format(3 * OUT_TAB, groups[1], '' if i == last - 1 else ',')
            body += '{:s}{:s}\n'.format(2 * OUT_TAB, lines[last].strip())
    if para_pth_path is not None:
        body += '{:s}self.load_state_dict(torch.load("{:s}"))\n'.format(2 * OUT_TAB, para_pth_path)
    funcs.insert(-1, 'nn.Flatten()')
    footer = '''{:s}def forward(self, x):\n{:s}{:s}return x\n'''.format(OUT_TAB, ''.join(['{:s}x = {:s}(x)\n'.format(2 * OUT_TAB, func) for func in funcs]), 2 * OUT_TAB)
    return header + body + footer