"""Conservative discovery of native reasoning controls from Jinja templates."""

from functools import lru_cache

from .output_parsers import REASONING_MARKERS


EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


@lru_cache(maxsize=128)
def template_controls(template):
    # All model template renderers depend on Jinja. Keep discovery optional for
    # services whose tokenizer does not use it, and never execute the template.
    try:
        from jinja2 import Environment, TemplateSyntaxError, meta, nodes
    except ImportError:
        return (), False, None
    try:
        tree = Environment(extensions=["jinja2.ext.loopcontrols", "jinja2.ext.do"]).parse(template)
        names = meta.find_undeclared_variables(tree)
    except TemplateSyntaxError:
        return (), False, None

    toggle = bool({"enable_thinking", "thinking"} & names)
    effort = "reasoning_effort" in names
    if not effort:
        return (), toggle, False

    aliases = {"reasoning_effort"}
    assignments = list(tree.find_all(nodes.Assign))
    # Follow local aliases such as Sharp's _effort_raw, rather than attributing
    # comparisons of message text or unrelated parameters to reasoning effort.
    for _ in range(len(assignments)):
        before = len(aliases)
        for assignment in assignments:
            if isinstance(assignment.target, nodes.Name) and any(
                name.name in aliases for name in assignment.node.find_all(nodes.Name)
            ):
                aliases.add(assignment.target.name)
            if (isinstance(assignment.target, nodes.Name)
                    and isinstance(assignment.node, nodes.Name) and assignment.node.name in aliases):
                aliases.add(assignment.target.name)
        if len(aliases) == before:
            break

    declared, canonical = set(), set()
    disabled = False
    for branch in tree.find_all(nodes.If):
        references = {name.name for name in branch.test.find_all(nodes.Name)}
        if isinstance(branch.test, nodes.Name):
            references.add(branch.test.name)
        if not references.intersection(aliases):
            continue
        comparisons = list(branch.test.find_all(nodes.Compare))
        if isinstance(branch.test, nodes.Compare):
            comparisons.append(branch.test)
        literals = set()
        for comparison in comparisons:
            for operand in comparison.ops:
                if operand.op in {"in", "notin"} and isinstance(operand.expr, (nodes.List, nodes.Tuple)):
                    literals.update(item.value for item in operand.expr.items
                                    if isinstance(item, nodes.Const) and isinstance(item.value, str))
                elif (operand.op in {"in", "notin"} and isinstance(operand.expr, nodes.Const)
                      and isinstance(operand.expr.value, (list, tuple))):
                    # Jinja's meta compiler can fold literal list/tuple nodes.
                    literals.update(item for item in operand.expr.value if isinstance(item, str))
                elif operand.op in {"eq", "ne"} and isinstance(operand.expr, nodes.Const):
                    if isinstance(operand.expr.value, str):
                        literals.add(operand.expr.value)
        declared.update(literals.intersection(EFFORTS))
        # A template can accept aliases but normalize them to fewer native
        # levels. Advertise those canonical levels, not each accepted spelling.
        for assignment in branch.find_all(nodes.Assign):
            target = assignment.target
            name = target.name if isinstance(target, nodes.Name) else (
                target.attr if isinstance(target, nodes.NSRef) else ""
            )
            if (isinstance(assignment.node, nodes.Const) and "effort" in name
                    and name not in aliases and assignment.node.value in EFFORTS):
                canonical.add(assignment.node.value)
        if "none" in literals:
            for statement in branch.body:
                if (isinstance(statement, nodes.Assign) and isinstance(statement.target, nodes.Name)
                        and "thinking" in statement.target.name
                        and isinstance(statement.node, nodes.Const) and statement.node.value is False):
                    disabled = True
    levels = canonical or declared
    if canonical and disabled:
        levels.add("none")
    return tuple(value for value in EFFORTS if value in levels), toggle, True


def reasoning_metadata(template, parser, defaults):
    if parser == "none":
        return {"supported": False}
    known = parser != "auto" or any(
        opening.rstrip() in template for _, opening, _ in REASONING_MARKERS
    )
    if not known:
        return None
    levels, toggle, effort = template_controls(template)
    result = {"supported": True}
    if levels:
        result.update(control="effort", supported_efforts=list(levels))
        default = defaults["reasoning_effort"] if defaults["enable_thinking"] else "none"
        if default in levels:
            result["default_effort"] = default
    elif toggle:
        result.update(control="toggle", default_enabled=defaults["enable_thinking"])
    elif effort is False:
        result["control"] = "fixed"
    # An effort variable without a verifiable level set is insufficient evidence
    # for a list or default. Leave its control unknown instead of inventing one.
    return result
