<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0-Prizren" styleCategories="Symbology|Labeling">
  <renderer-v2 type="RuleRenderer" symbollevels="0" enableorderby="0" forceraster="0" referencescale="-1">
    <rules key="{root}">
      <rule key="{r0}" label="Out of service" symbol="0" filter="&quot;status&quot; &lt;&gt; 'in_operation'" />
      <rule key="{r1}" label="HVDC" symbol="1" filter="&quot;hvdc&quot; = 1" />
      <rule key="{r2}" label="Interconnector (cross-border)" symbol="2" filter="&quot;status&quot; = 'in_operation' AND &quot;hvdc&quot; = 0 AND &quot;is_interconnector&quot; = 1" />
      <rule key="{r3}" label="50Hertz" symbol="3" filter="&quot;status&quot; = 'in_operation' AND &quot;hvdc&quot; = 0 AND &quot;is_interconnector&quot; = 0 AND &quot;tso&quot; = '50Hertz'" />
      <rule key="{r4}" label="Amprion" symbol="4" filter="&quot;status&quot; = 'in_operation' AND &quot;hvdc&quot; = 0 AND &quot;is_interconnector&quot; = 0 AND &quot;tso&quot; = 'Amprion'" />
      <rule key="{r5}" label="TenneT" symbol="5" filter="&quot;status&quot; = 'in_operation' AND &quot;hvdc&quot; = 0 AND &quot;is_interconnector&quot; = 0 AND &quot;tso&quot; = 'TenneT'" />
      <rule key="{r6}" label="TransnetBW" symbol="6" filter="&quot;status&quot; = 'in_operation' AND &quot;hvdc&quot; = 0 AND &quot;is_interconnector&quot; = 0 AND &quot;tso&quot; = 'TransnetBW'" />
    </rules>
    <symbols>
      <symbol type="line" name="0" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="158,158,158,255" />
            <Option name="line_width" type="QString" value="0.4" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="dot" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="offset" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(coalesce(&quot;circuit_no&quot;, 0) - (coalesce(&quot;circuits&quot;, 1) - 1) / 2.0) * 0.8" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="line" name="1" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="33,102,172,255" />
            <Option name="line_width" type="QString" value="0.9" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="1" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
            <Option name="customdash" type="QString" value="3;1.6" />
            <Option name="customdash_unit" type="QString" value="MM" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="offset" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(coalesce(&quot;circuit_no&quot;, 0) - (coalesce(&quot;circuits&quot;, 1) - 1) / 2.0) * 0.8" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="line" name="2" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="255,255,255,255" />
            <Option name="line_width" type="QString" value="1.6" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="outlineWidth" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="CASE WHEN &quot;voltage_kv&quot; &gt;= 300 THEN 0.6 ELSE 0.4 END + 0.9" />
                  <Option name="type" type="int" value="3" />
                </Option>
                <Option name="offset" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(coalesce(&quot;circuit_no&quot;, 0) - (coalesce(&quot;circuits&quot;, 1) - 1) / 2.0) * 0.8" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="106,61,154,255" />
            <Option name="line_width" type="QString" value="0.8" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="outlineWidth" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="CASE WHEN &quot;voltage_kv&quot; &gt;= 300 THEN 0.6 ELSE 0.4 END" />
                  <Option name="type" type="int" value="3" />
                </Option>
                <Option name="offset" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(coalesce(&quot;circuit_no&quot;, 0) - (coalesce(&quot;circuits&quot;, 1) - 1) / 2.0) * 0.8" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="line" name="3" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="255,255,255,230" />
            <Option name="line_width" type="QString" value="1.1" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="outlineWidth" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="CASE WHEN &quot;voltage_kv&quot; &gt;= 300 THEN 0.6 ELSE 0.4 END + 0.45" />
                  <Option name="type" type="int" value="3" />
                </Option>
                <Option name="offset" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(coalesce(&quot;circuit_no&quot;, 0) - (coalesce(&quot;circuits&quot;, 1) - 1) / 2.0) * 0.8" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="42,120,214,255" />
            <Option name="line_width" type="QString" value="0.7" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="outlineWidth" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="CASE WHEN &quot;voltage_kv&quot; &gt;= 300 THEN 0.6 ELSE 0.4 END" />
                  <Option name="type" type="int" value="3" />
                </Option>
                <Option name="offset" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(coalesce(&quot;circuit_no&quot;, 0) - (coalesce(&quot;circuits&quot;, 1) - 1) / 2.0) * 0.8" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="line" name="4" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="255,255,255,230" />
            <Option name="line_width" type="QString" value="1.1" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="outlineWidth" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="CASE WHEN &quot;voltage_kv&quot; &gt;= 300 THEN 0.6 ELSE 0.4 END + 0.45" />
                  <Option name="type" type="int" value="3" />
                </Option>
                <Option name="offset" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(coalesce(&quot;circuit_no&quot;, 0) - (coalesce(&quot;circuits&quot;, 1) - 1) / 2.0) * 0.8" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="235,104,52,255" />
            <Option name="line_width" type="QString" value="0.7" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="outlineWidth" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="CASE WHEN &quot;voltage_kv&quot; &gt;= 300 THEN 0.6 ELSE 0.4 END" />
                  <Option name="type" type="int" value="3" />
                </Option>
                <Option name="offset" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(coalesce(&quot;circuit_no&quot;, 0) - (coalesce(&quot;circuits&quot;, 1) - 1) / 2.0) * 0.8" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="line" name="5" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="255,255,255,230" />
            <Option name="line_width" type="QString" value="1.1" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="outlineWidth" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="CASE WHEN &quot;voltage_kv&quot; &gt;= 300 THEN 0.6 ELSE 0.4 END + 0.45" />
                  <Option name="type" type="int" value="3" />
                </Option>
                <Option name="offset" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(coalesce(&quot;circuit_no&quot;, 0) - (coalesce(&quot;circuits&quot;, 1) - 1) / 2.0) * 0.8" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="27,175,122,255" />
            <Option name="line_width" type="QString" value="0.7" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="outlineWidth" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="CASE WHEN &quot;voltage_kv&quot; &gt;= 300 THEN 0.6 ELSE 0.4 END" />
                  <Option name="type" type="int" value="3" />
                </Option>
                <Option name="offset" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(coalesce(&quot;circuit_no&quot;, 0) - (coalesce(&quot;circuits&quot;, 1) - 1) / 2.0) * 0.8" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="line" name="6" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="255,255,255,230" />
            <Option name="line_width" type="QString" value="1.1" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="outlineWidth" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="CASE WHEN &quot;voltage_kv&quot; &gt;= 300 THEN 0.6 ELSE 0.4 END + 0.45" />
                  <Option name="type" type="int" value="3" />
                </Option>
                <Option name="offset" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(coalesce(&quot;circuit_no&quot;, 0) - (coalesce(&quot;circuits&quot;, 1) - 1) / 2.0) * 0.8" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="237,161,0,255" />
            <Option name="line_width" type="QString" value="0.7" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="outlineWidth" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="CASE WHEN &quot;voltage_kv&quot; &gt;= 300 THEN 0.6 ELSE 0.4 END" />
                  <Option name="type" type="int" value="3" />
                </Option>
                <Option name="offset" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(coalesce(&quot;circuit_no&quot;, 0) - (coalesce(&quot;circuits&quot;, 1) - 1) / 2.0) * 0.8" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="rule-based">
    <rules key="{labels-root}">
      <rule key="{label-0}" description="Line name (one per circuit)" scalemaxdenom="550000" scalemindenom="1">
        <settings calloutType="simple">
          <text-style fieldName="&quot;name&quot;" isExpression="1" fontFamily="Sans Serif" fontSize="7.0" fontSizeUnit="Point" fontWeight="50" textColor="33,33,33,255" textOpacity="1" fontItalic="0" namedStyle="Regular" multilineHeight="1" allowHtml="0">
            <text-buffer bufferDraw="1" bufferSize="0.8" bufferSizeUnits="MM" bufferColor="255,255,255,230" bufferOpacity="0.9" bufferJoinStyle="128" bufferNoFill="1" />
          </text-style>
          <text-format wrapChar="" autoWrapLength="0" multilineAlign="0" plussign="0" reverseDirectionSymbol="0" />
          <placement placement="2" placementFlags="10" dist="1.2" distUnits="MM" priority="5" repeatDistance="0" overrunDistance="0" offsetType="0" quadOffset="4" centroidInside="1" fitInPolygonOnly="0" />
          <rendering obstacle="1" obstacleFactor="1" obstacleType="1" upsidedownLabels="0" displayAll="0" mergeLines="1" labelPerPart="0" limitNumLabels="0" maxNumLabels="2000" fontLimitPixelSize="0" scaleVisibility="0" drawLabels="1" />
        </settings>
      </rule>
      <rule key="{label-1}" description="Circuits in the corridor" scalemaxdenom="4500000" scalemindenom="1" filter="&quot;circuit_no&quot; = 0 AND &quot;circuits&quot; &gt;= 2 AND @map_scale &gt; 550000">
        <settings calloutType="simple">
          <text-style fieldName="&quot;circuits&quot; || '×'" isExpression="1" fontFamily="Sans Serif" fontSize="7.0" fontSizeUnit="Point" fontWeight="50" textColor="33,33,33,255" textOpacity="1" fontItalic="0" namedStyle="Regular" multilineHeight="1" allowHtml="0">
            <text-buffer bufferDraw="1" bufferSize="0.8" bufferSizeUnits="MM" bufferColor="255,255,255,230" bufferOpacity="0.9" bufferJoinStyle="128" bufferNoFill="1" />
          </text-style>
          <text-format wrapChar="" autoWrapLength="0" multilineAlign="0" plussign="0" reverseDirectionSymbol="0" />
          <placement placement="2" placementFlags="10" dist="1.2" distUnits="MM" priority="5" repeatDistance="0" overrunDistance="0" offsetType="0" quadOffset="4" centroidInside="1" fitInPolygonOnly="0" />
          <rendering obstacle="1" obstacleFactor="1" obstacleType="1" upsidedownLabels="0" displayAll="0" mergeLines="1" labelPerPart="0" limitNumLabels="0" maxNumLabels="2000" fontLimitPixelSize="0" scaleVisibility="0" drawLabels="1" />
        </settings>
      </rule>
    </rules>
  </labeling>
  <layerGeometryType>1</layerGeometryType>
</qgis>
