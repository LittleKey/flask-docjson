# -*- coding: utf-8 -*-

"""
    flask-docjson
    ~~~~~~~~~~~~~

    Validate flask request via docstring.

    :copyright: (c) 2016 by Chao Wang (hit9).
    :license: BSD, see LICENSE for more details.
"""

import sys
import threading


###
# Exceptions
###

class Error(Exception):
    """A flask-docjson error occurred."""


class ParserError(Error):
    """A parser error occurred.

    Catching this error will catch both
    :exc:`~flask_docjson.LexerError` and
    :exc:`~flask_docjson.ParserError` errors.
    """


class LexerError(ParserError):
    """A lexer error occurred."""


class GrammarError(ParserError):
    """A parser error occurred."""


class _InternalError(Exception):
    """Internal purpose error. It will be replaced by
    :exc:`~flask_docjson.Error` after raised.
    """


class _InternalLexerError(_InternalError):
    """An internal lexer error occurred."""


class _InternalGrammarError(_InternalGrammarError):
    """An internal parser error occurred."""


###
# Compact
###

def get_func_code(func):
    """Returns the ``code`` object of given flask view ``func``.
    """
    if sys.version_info.major == 3:  # Py3
        return func.__code__
    return func.func_code  # Py2


###
# Globals
###

S_ELLIPSIS = 0  # Ellipsis is ``...``

# Base types
T_BOOL = 1
T_U8 = 2
T_U16 = 3
T_U32 = 4
T_U64 = 5
T_I8 = 6
T_I16 = 7
T_I32 = 8
T_I64 = 9
T_FLOAT = 10
T_STRING = 11

# HTTP methods
M_POST = 1
M_GET = 2
M_PUT = 3
M_DELETE = 4
M_PATCH = 5
M_HEAD = 6
M_OPTIONS = 7


###
# Lexer
###

literals = ':,()[]{}/<>*?=&'

t_ignore = ' \t\r'   # Ignore white spaces

tokens = (
    # HTTP methods
    'POST',
    'GET',
    'PUT',
    'DELETE',
    'PATCH',
    'HEAD',
    'OPTIONS',
    # Ellipsis
    'ELLIPSIS',
    # Base types
    'BOOL',
    'U8',
    'U16',
    'U32',
    'U64',
    'I8',
    'I16',
    'I32',
    'I64',
    'FLOAT',
    'STRING',
    # Others
    'IDENTIFIER',
    'STATIC_ROUTE',
    'STATUS_CODE_MATCHER',
    'LITERAL_INTEGER',
    'LITERAL_STRING',
    'AS',
)


def t_error(t):
    raise _InternalLexerError("illegal char '{0}' at line {1}".format(
        t.value[0], t.lineno))


def t_ignore_COMMENT(t):
    r'\/\/[^\n]*'
    # Ignore comments, e.g.: ``// this is an example comment``


def t_newline(t):
    r'\n+'
    # Count newline to ``lexer.lineno``
    t.lexer.lineno += len(t.value)


def t_POST(t):
    r'POST'
    # HTTP method ``POST``
    t.value = M_POST
    return t


def t_GET(t):
    r'GET'
    # HTTP method ``GET``
    t.value = M_GET
    return t


def t_PUT(t):
    r'PUT'
    # HTTP method ``PUT``
    t.value = M_PUT
    return t


def t_DELETE(t):
    r'DELETE'
    # HTTP method ``DELETE``
    t.value = M_DELETE
    return t


def t_HEAD(t):
    r'HEAD'
    # HTTP method ``HEAD``
    t.value = M_HEAD
    return t


def t_OPTIONS(t):
    r'OPTIONS'
    # HTTP method ``OPTIONS``
    t.value = M_OPTIONS
    return


def t_PATCH(t):
    r'PATCH'
    # HTTP method ``PATCH``
    t.value = M_PATCH
    return t


def t_ELLIPSIS(t):
    r'\.\.\.'
    # Ellipsis: ``...``
    t.value = S_ELLIPSIS
    return t


def t_BOOL(t):
    r'bool'
    t.value = T_BOOL
    return t


def t_U8(t):
    r'u8'
    t.value = T_U8
    return t


def t_U16(t):
    r'u16'
    t.value = T_U16
    return t


def t_U32(t):
    r'u32'
    t.value = T_U32
    return t


def t_U64(t):
    r'u64'
    t.value = T_U64
    return t


def t_I8(t):
    r'i8'
    t.value = T_I8
    return t


def t_I16(t):
    r'i16'
    t.value = T_I16
    return t


def t_I32(t):
    r'i32'
    t.value = T_I32
    return t


def t_I64(t):
    r'i64'
    t.value = T_I64
    return t


def t_FLOAT(t):
    r'float'
    t.value = T_FLOAT
    return t


def t_STRING(t):
    r'string'
    t.value = T_STRING
    return t


def t_AS(t):
    r'as'
    return t


def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    # Identifier, e.g. ``word``, ``word1``.
    return t


def t_STATIC_ROUTE(t):
    r'\B/[^<{\r\n\s]*'
    # Static route, e.g.:``/``, ``/order/``
    return t


def t_STATUS_CODE_MATCHER(t):
    r'[0-9]+[X]+'
    # Status code matcher, e.g.: ``4XX``
    return t


def t_LITERAL_INTEGER(t):
    r'[+-]?[0-9]+'
    # Integer literal, e.g.: ``404``, ``200``, ``201``
    t.value = int(t.value)
    return t


def t_LITERAL_STRING(t):
    r'(\"([^\\\n]|(\\.))*?\")'
    # String literal, e.g.: ``"string"``
    s = t.value[1:-1]  # Get content inside ``""``
    # Translate escaping chars.
    # Cause: original chars are actually two, e.g. ``"\\t"``,
    # we are going to translate them into a single char ``"\t"``.
    maps = {  # Escaping char we support
        't': '\t',
        'r': '\r',
        'n': '\n',
        '\\': '\\',
        '"': '\"'
    }
    i = 0
    length = len(s)
    val = ''
    while i < length:
        if s[i] == '\\':  # Escaping leader char
            i += 1
            if s[i] in maps:
                val += maps[s[i]]
            else:
                raise _InternalLexerError("unsupported escaping char '{0}' at "
                                          "{1}".format(s[i], t.lineno))
        else:  # Non-escaping/normal char
            val += s[i]
        i += 1
    t.value = val
    return t


###
# Parser
###

class Schema(object):
    """Full schema parse result.

    :param request: The parsed :class:`Request <Request>`.
    :param responses: The list of parsed :class:`Response <Response>`.
    """

    def __init__(self, request, responses):
        self.request = request
        self.responses = responses


class Request(object):
    """The request parse result.

    :param methods: The list of parsed http method codes.
    :param route: The parsed :class:`Route <Route>`.
    :param json_schema: (optional) The parsed :class:`JsonSchema <JsonSchema>`.
    """
    def __init__(self, methods, route, json_schema=None):
        self.methods = methods
        self.route = route
        self.json_schema = json_schema

    @property
    def methods_repr(self):
        return '/'.join(self.methods)

    def __repr__(self):
        return '<Request [{0} {1}]>'.format(self.methods_repr, self.route.rule)


class Response(object):
    """The response parse result.

    :param status_codes: The list of parsed status codes and status code
       matchers.
    :param json_schema: (optional) The parsed :class:`JsonSchema <JsonSchema>`.
    """
    def __init__(self, status_codes, json_schema=None):
        self.status_codes = status_codes
        self.json_schema = json_schema

    @property
    def status_codes_repr(self):
        return '/'.join(map(str, self.status_codes))

    def __repr__(self):
        return '<Response [{0}]>'.format(self.status_codes_repr)


class Route(object):
    """The route parse result.
    """
    def __init__(self):
        # The string route rule. e.g.: ``'/order/<i32>'``
        self.rule = None
        # Optional dictionary of url variable name to type pairs.
        # e.g.: ``{'id': <I32, required>}``
        self.url_variables = None
        # Optional dictionary of url parameters name to type pairs.
        self.url_parameters = None

    def __repr__(self):
        return '<Route [{0}]>'.format(self.rule)

    def add_url_variable(self, static_rule, name=None, typ=None):
        """Add an `url_variable` item to this route object.
        This method should not be called from user code, it's made for parsing.

        :param static_rule: The static route rule string.
        :param name: A string represents the url variable name.
        :param typ: The parsed :class:`Type <Type>` of this url variable.
        """
        # Reconstruct ``self.rule``.
        rule = static_rule
        if name:
            if typ:
                rule += '<{0}:{1}>'.format(typ, name)
            else:
                rule += '<{0}>'.format(name)
        self.rule = rule
        # Add ``name:typ`` to ``self.url_variables``.
        if name:
            if self.url_variables is None:
                self.url_variables = {}
            self.url_variables[name] = typ

    def add_url_parameter(self, name, typ):
        """Add an `url_parameter` item to this route object.
        This method should not be called from user code, it's made for parsing.

        :param name: The url parameter name.
        :param typ: The parsed :class:`Type <Type>` of this url parameter.
        """
        if self.url_parameters is None:
            self.url_parameters = {}
        self.url_parameters[name] = typ


class JsonSchema(object):
    """The json_schema parse result.

    :param data: The original json schema data.
    :param has_ellipsis: A boolean value indicates whether there is
       ``ELLIPSIS`` sign.
    """
    def __init__(self, data, has_ellipsis=False):
        self.data = data
        self.has_ellipsis = has_ellipsis

    def is_array(self):
        """Returns ``True`` if this json_schema is an array.
        """
        return isinstance(self.data, list)

    def is_object(self):
        """Returns ``True`` if this json_schema is an object.
        """
        return isinstance(self.data, dict)


class Type(object):
    """The type parse result.

    :param base: The base type to construct the :class:`Type <Type>`.
    :param required: A boolean value indicates whether value can be optional or
       ``None``, defaults to ``True``.
    """
    def __init__(self, base, required=True, is_ref_type=False):
        if isinstance(base, Type):
            # Flat ``base`` if it's a ``Type``.
            self.base = base.base
        else:
            self.base = base
        self.required = required
        self.is_ref_type = is_ref_type
        # The original type of a ref_type if `is_ref_type` is set.
        self.orig_type = None

    def is_string(self):
        """Returns ``True`` if this type is a string type.
        """
        if isinstance(self.base, tuple):
            if self.base[0] == T_STRING:
                return True
        return False

    def get_string_length(self):
        """Returns string length constraint value if this type is a string
        type. Otherwise returns ``None``.
        """
        if self.is_string():
            return self.base[1]

    def is_base_type(self):
        """Returns ``True`` if this type is a base type.
        """
        if self.base in (T_BOOL,
                         T_U8,
                         T_U16,
                         T_U32,
                         T_U64,
                         T_I8,
                         T_I16,
                         T_I32,
                         T_I64,
                         T_FLOAT):
            return True
        if self.is_string():
            return True
        return False


class TypeReference(object):
    """Reference abstraction for ref_type, used to hold the reference context
    such as ``lineno`` etc.

    :param name: The type name of the reference.
    :parm typ: The generated ``ref_type`` for this reference.
    :param lineno: The line number where the reference locates.
    """
    def __init__(self, name, typ, lineno=None, func=None):
        self.name = name
        self.typ = typ
        self.lineno = lineno
        self.func = func


class ParseContext(threading.local):
    """Global parsing context to hold runtime data. This should not be used
    from user code, it's made for parsing. Also global variables sucks, but
    ply library requires that.

    The constructed context is a threading local object to avoid unsafe
    behaviors on multiple threading environments such as gunicorn workers. The
    one thing only to ensure is that we should patch ``threading`` with gevent
    before ``flask-docjson`` is loaded, if you are using gevent workers.

    The lifetime of this context is from full flask app parsing start to end.
    """
    def __init__(self):
        # A map of `name` to `type` for ref_type
        self.ref_type_map = {}
        # A list of references for ref_type
        self.references = []
        # Current working on view_function.
        self.current_func = None

    def set_current_func(self, func):
        """Set current parsing view function.
        """
        self.current_func = func

    def register_ref_type(self, name, typ, lineno=None):
        """Register a ref_type to the context.
        """
        if name in self.ref_type_map:
            exc = _InternalGrammarError(
                "Duplicated type reference definition: "
                "'{0}' at line {1}".format(name, lineno))
            raise_parse_error(self.current_func, exc)
        self.ref_type_map[name] = typ

    def append_reference(self, reference):
        """Record a type reference.
        """
        self.references.append(reference)

    def resolve_references(self):
        """Resolve all references to types, this will replace the ``ref_type``
        with the referenced original type.

        Raises :class:`GrammarError <GrammarError>` if any referenced type
        is not found.

        This should be called after all view functions are parsed.
        """
        for ref in self.references:
            if ref.name not in self.ref_type_map:
                exc = _InternalGrammarError(
                    "Undefined type alias: "
                    "'{0}' at line {1}".format(ref.name,
                                               ref.lineno))
                raise_parse_error(ref.func, exc)
            ref.typ.orig_type = self.ref_type_map[ref.name]


# Global threading-local parse context.
_parse_ctx = ParseContext()


def raise_parse_error(func, exc):
    """Format parse exception ``exc`` with flask view function ``func`` and
    raise it.

    :param func: A flask view function.
    :param exc: An internal parse exception, instance of
       :class:`_InternalLexerError <_InternalLexerError>`,
       or :class:`_InternalGrammarError <_InternalGrammarError>`,
       or :class:`_InternalError <_InternalError>`.
    """
    func_code = get_func_code(func)
    msg = ('An error occurred on function {0} (defined at {1}:{2}), original '
           'exception on its schema definition is:{3}')\
        .format(func_code.co_name,
                func_code.co_filename,
                func_code.co_firstlineno,
                exc)
    if isinstance(exc, _InternalLexerError):
        raise LexerError(msg)
    elif isinstance(exc, _InternalGrammarError):
        raise GrammarError(msg)
    else:
        raise ParserError(msg)


def _parse_seq(p):
    """Util function to parse recursive sequence::

        def p_seq(p):
            '''seq : item seq
                   | item
                   |'''
            _parse_seq(p)

    """
    if len(p) == 4:
        p[0] = [p[1]] + p[3]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]
    elif len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 1:
        p[0] = []


def p_error(p):
    if p is None:
        raise _InternalGrammarError("grammar error at EOF")
    raise _InternalGrammarError("grammar error '{0}' at line {1}".format(
        p.value, p.lineno))


def p_start(p):
    '''start : request response_seq'''
    p[0] = Schema(p[1], p[2])


def p_request(p):
    '''request : method_seq route json_schema
               | method_seq route'''
    if len(p) == 4:  # With json schema
        p[0] = Request(p[1], p[2], p[3])
    elif len(p) == 3:  # No json schema
        p[0] = Request(p[1], p[2])


def p_method_seq(p):
    '''method_seq : method '/' method_seq
                  | method method_seq
                  |'''
    _parse_seq(p)


def p_method(p):
    '''method : POST
              | GET
              | PUT
              | DELETE
              | PATCH
              | HEAD
              | OPTIONS'''
    p[0] = p[1]


def p_route(p):
    '''route : route route_item url_parameters
             | route route_item
             |'''
    if len(p) == 1:
        p[0] = Route()
    else:
        route = p[1]
        static_rule = p[2][0]
        if p[2][1]:
            url_variable_name, url_variable_type = p[2][1]
        else:
            url_variable_name, url_variable_type = None, None
        # Add url variable
        route.add_url_variable(static_rule, url_variable_name,
                               url_variable_type)
        if len(p) == 4:
            # Add url parameters
            for url_parameter_name, url_parameter_type in p[3]:
                route.add_url_parameter(url_parameter_name, url_parameter_type)
        p[0] = route


def p_route_item(p):
    '''route_item : STATIC_ROUTE url_variable
                  | STATIC_ROUTE'''
    # Returns a tuple in form of ``(static, url_variable)``.
    # Where ``url_variable`` is optional, default ``None``.
    if len(p) == 3:
        p[0] = (p[1], p[2])
    elif len(p) == 2:
        p[0] = (p[1], None)


def p_url_variable(p):
    '''url_variable : '<' base_type ':' IDENTIFIER '>'
                    | '<' IDENTIFIER '>' '''
    # Returns a tuple in form of ``(name, type)``.
    # Where ``type`` is optional, default ``None``.
    # Note that ``type`` must be a required base type.
    if len(p) == 6:
        p[0] = (p[4], p[2])
    elif len(p) == 4:
        p[0] = (p[2], None)


def p_url_parameters(p):
    '''url_parameters : '?' url_parameter_seq '''
    p[0] = p[2]


def p_url_parameter_seq(p):
    '''url_parameter_seq : url_parameter_item '&' url_parameter_seq
                         | url_parameter_item
                         |'''
    _parse_seq(p)


def p_url_parameter_item(p):
    '''url_parameter_item : IDENTIFIER '=' base_type_may_optional '''
    # Returns a tuple in form of ``(name, type)``.
    p[0] = (p[1], [3])


def p_response_seq(p):
    '''response_seq : response_item response_seq
                    | response_item'''
    # Returns list of :class:`Response <Response>`.
    # A schema must have at least one response.
    _parse_seq(p)


def p_response_item(p):
    '''response_item : status_code_seq json_schema
                     | status_code_seq'''
    if len(p) == 3:
        p[0] = Response(p[1], p[2])
    elif len(p) == 2:
        p[0] = Response(p[1])


def p_status_code_seq(p):
    '''status_code_seq : status_code_item '/' status_code_seq
                       | status_code_item'''
    # A response must have at least one status code.
    _parse_seq(p)


def p_status_code_item(p):
    '''status_code_item : STATUS_CODE_MATCHER
                        | LITERAL_INTEGER'''
    p[0] = p[1]


def p_json_schema(p):
    '''json_schema : json_object
                   | json_array '''
    p[0] = p[1]


def p_json_object(p):
    '''json_object : '{' json_kv_seq '}'
                   | '{' json_kv_seq ELLIPSIS '}' '''
    p[0] = JsonSchema(dict(p[2]), has_ellipsis=(len(p) == 5))


def p_json_array(p):
    '''json_array : '[' json_value_seq ']'
                  | '[' json_value_seq ELLIPSIS ']' '''
    p[0] = JsonSchema(p[2], has_ellipsis=(len(p) == 5))


def p_json_kv_seq(p):
    '''json_kv_seq : json_kv_item ',' json_kv_seq
                   | json_kv_item
                   |'''
    _parse_seq(p)


def p_json_kv(p):
    '''json_kv_item : LITERAL_STRING ':' json_value '''
    p[0] = (p[1], p[3])


def p_json_value_seq(p):
    '''json_value_seq : json_value ',' json_value_seq
                      | json_value
                      |'''
    _parse_seq(p)


def p_json_value(p):
    '''json_value : type
                  | simple_type_as_ref'''
    p[0] = p[1]


def p_type(p):
    '''type : simple_type_may_optional
            | ref_type_may_optional'''
    p[0] = p[1]


def p_simple_type_as_ref(p):
    '''simple_type_as_ref : simple_type_may_optional AS IDENTIFIER '''
    p[0] = p[1]
    # Register referenced type
    _parse_ctx.register_ref_type(p[3], p[1], p.lineno)


def p_ref_type_may_optional(p):
    '''ref_type_may_optional : ref_type
                             | ref_type '*' '''
    p[0] = Type(p[1], reuired=(len(p) == 2))


def p_ref_type(p):
    '''ref_type : IDENTIFIER'''
    # Generate a ``type`` for this ref_type
    ref_typ = Type(p[1], is_ref_type=True)
    p[0] = ref_typ
    # Record this reference for later replacement.
    reference = TypeReference(p[1], ref_typ,
                              lineno=p.lineno,
                              func=_parse_ctx.current_func)
    _parse_ctx.append_reference(reference)


def p_simple_type_may_optional(p):
    '''simple_type_may_optional : simple_type
                                | simple_type '*' '''
    p[0] = Type(p[1], reuired=(len(p) == 2))


def p_simple_type(p):
    '''simple_type : base_type
                   | json_schema'''
    p[0] = Type(p[1])


def p_base_type_may_optional(p):
    '''base_type_may_optional : base_type
                              | base_type '*' '''
    p[0] = Type(p[1], reuired=(len(p) == 2))


def p_base_type(p):
    '''base_type : BOOL
                 | U8
                 | U16
                 | U32
                 | U64
                 | I8
                 | I16
                 | I32
                 | I64
                 | FLOAT
                 | string_type'''
    p[0] = Type([1])


def p_string_type(p):
    '''string_type : STRING
                   | STRING '(' LITERAL_INTEGER ')' '''
    if len(p) == 2:
        p[0] = Type((p[1], None))
    elif len(p) == 5:
        p[0] = Type((p[1], p[3]))
