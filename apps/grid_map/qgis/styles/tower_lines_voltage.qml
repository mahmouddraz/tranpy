<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0-Prizren" styleCategories="Symbology|Labeling">
  <renderer-v2 type="RuleRenderer" symbollevels="0" enableorderby="0" forceraster="0" referencescale="-1">
    <rules key="{root}">
      <rule key="{r0}" label="Tower line (OpenStreetMap)" symbol="0" scalemaxdenom="1100000" scalemindenom="1" />
    </rules>
    <symbols>
      <symbol type="line" name="0" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="158,158,158,217" />
            <Option name="line_width" type="QString" value="0.35" />
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
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="rule-based">
    <rules key="{labels-root}">
      <rule key="{label-0}" description="Tower line ref" scalemaxdenom="140000" scalemindenom="1" filter="&quot;ref&quot; &lt;&gt; ''">
        <settings calloutType="simple">
          <text-style fieldName="&quot;ref&quot;" isExpression="1" fontFamily="Sans Serif" fontSize="6.0" fontSizeUnit="Point" fontWeight="50" textColor="33,33,33,255" textOpacity="1" fontItalic="0" namedStyle="Regular" multilineHeight="1" allowHtml="0">
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
